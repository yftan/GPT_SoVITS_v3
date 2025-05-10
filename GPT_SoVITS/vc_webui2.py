'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
import logging
import traceback,torchaudio,warnings
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)

import os, re, sys, json
import pdb
import torch
from text.LangSegmenter import LangSegmenter
import torch.nn.functional as F

try:
    import gradio.analytics as analytics
    analytics.version_check = lambda:None
except:...
version=model_version="v3"
pretrained_sovits_name=["GPT_SoVITS/pretrained_models/s2Gv3.pth"]
pretrained_gpt_name=["GPT_SoVITS/pretrained_models/s1v3.ckpt"]


_ =[[],[]]
for i in range(1):
    if os.path.exists(pretrained_gpt_name[i]):_[0].append(pretrained_gpt_name[i])
    if os.path.exists(pretrained_sovits_name[i]):_[-1].append(pretrained_sovits_name[i])
pretrained_gpt_name,pretrained_sovits_name = _


if os.path.exists(f"./weight.json"):
    pass
else:
    with open(f"./weight.json", 'w', encoding="utf-8") as file:json.dump({'GPT':{},'SoVITS':{}},file)

with open(f"./weight.json", 'r', encoding="utf-8") as file:
    weight_data = file.read()
    weight_data=json.loads(weight_data)
    gpt_path = os.environ.get(
        "gpt_path", weight_data.get('GPT',{}).get(version,pretrained_gpt_name))
    sovits_path = os.environ.get(
        "sovits_path", weight_data.get('SoVITS',{}).get(version,pretrained_sovits_name))
    if isinstance(gpt_path,list):
        gpt_path = gpt_path[0]
    if isinstance(sovits_path,list):
        sovits_path = sovits_path[0]

# gpt_path = os.environ.get(
#     "gpt_path", pretrained_gpt_name
# )
# sovits_path = os.environ.get("sovits_path", pretrained_sovits_name)
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)
infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
punctuation = set(['!', '?', '…', ',', '.', '-'," "])
import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from feature_extractor import cnhubert

cnhubert.cnhubert_base_path = cnhubert_base_path

from GPT_SoVITS.module.models import SynthesizerTrn,SynthesizerTrnV3
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from tools.my_utils import load_audio
from tools.i18n.i18n import I18nAuto, scan_language_list

language=os.environ.get("language","Auto")
language=sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

dict_language_v1 = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
}
dict_language_v2 = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("粤语"): "all_yue",#全部按中文识别
    i18n("韩文"): "all_ko",#全部按韩文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("粤英混合"): "yue",#按粤英混合识别####不变
    i18n("韩英混合"): "ko",#按韩英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
    i18n("多语种混合(粤语)"): "auto_yue",#多语种启动切分识别语种
}
dict_language = dict_language_v1 if version =='v1' else dict_language_v2

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

resample_transform_dict={}
def resample(audio_tensor, sr0):
    global resample_transform_dict
    if sr0 not in resample_transform_dict:
        resample_transform_dict[sr0] = torchaudio.transforms.Resample(
            sr0, 24000
        ).to(device)
    return resample_transform_dict[sr0](audio_tensor)

def change_sovits_weights(sovits_path,prompt_language=None,text_language=None):
    global vq_model, hps, version, model_version, dict_language
    '''
        v1:about 82942KB
        half thr:82978KB
        v2:about 83014KB
        half thr:100MB
        v1base:103490KB
        half thr:103520KB
        v2base:103551KB
        v3:about 750MB
        
        ~82978K~100M~103420~700M
        v1-v2-v1base-v2base-v3
        version:
            symbols version and timebre_embedding version
        model_version:
            sovits is v1/2 (VITS) or v3 (shortcut CFM DiT)
    '''
    size=os.path.getsize(sovits_path)
    if size<82978*1024:
        model_version=version="v1"
    elif size<100*1024*1024:
        model_version=version="v2"
    elif size<103520*1024:
        model_version=version="v1"
    elif size<700*1024*1024:
        model_version = version = "v2"
    else:
        version = "v2"
        model_version="v3"

    dict_language = dict_language_v1 if version =='v1' else dict_language_v2
    if prompt_language is not None and text_language is not None:
        if prompt_language in list(dict_language.keys()):
            prompt_text_update, prompt_language_update = {'__type__':'update'},  {'__type__':'update', 'value':prompt_language}
        else:
            prompt_text_update = {'__type__':'update', 'value':''}
            prompt_language_update = {'__type__':'update', 'value':i18n("中文")}
        if text_language in list(dict_language.keys()):
            text_update, text_language_update = {'__type__':'update'}, {'__type__':'update', 'value':text_language}
        else:
            text_update = {'__type__':'update', 'value':''}
            text_language_update = {'__type__':'update', 'value':i18n("中文")}
        if model_version=="v3":
            visible_sample_steps=True
            visible_inp_refs=False
        else:
            visible_sample_steps=False
            visible_inp_refs=True
        yield  {'__type__':'update', 'choices':list(dict_language.keys())}, {'__type__':'update', 'choices':list(dict_language.keys())}, prompt_text_update, prompt_language_update, text_update, text_language_update,{"__type__": "update", "visible": visible_sample_steps},{"__type__": "update", "visible": visible_inp_refs},{"__type__": "update", "value": False,"interactive":True if model_version!="v3"else False}

    dict_s2 = torch.load(sovits_path, map_location="cpu", weights_only=False)
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version=hps.model.version
    # print("sovits版本:",hps.model.version)
    if model_version!="v3":
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        model_version=version
    else:
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
    if ("pretrained" not in sovits_path):
        try:
            del vq_model.enc_q
        except:pass
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print("loading sovits_%s"%model_version,vq_model.load_state_dict(dict_s2["weight"], strict=False))
    with open("./weight.json")as f:
        data=f.read()
        data=json.loads(data)
        data["SoVITS"][version]=sovits_path
    with open("./weight.json","w")as f:f.write(json.dumps(data))


try:next(change_sovits_weights(sovits_path))
except:pass

def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    # total = sum([param.nelement() for param in t2s_model.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./weight.json")as f:
        data=f.read()
        data=json.loads(data)
        data["GPT"][version]=gpt_path
    with open("./weight.json","w")as f:f.write(json.dumps(data))


change_gpt_weights(gpt_path)
os.environ["HF_ENDPOINT"]          = "https://hf-mirror.com"
import torch,soundfile
now_dir = os.getcwd()
import soundfile

def init_bigvgan():
    global model
    from BigVGAN import bigvgan
    model = bigvgan.BigVGAN.from_pretrained("%s/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x" % (now_dir,), use_cuda_kernel=False)  # if True, RuntimeError: Ninja is required to load C++ extensions
    # remove weight norm in the model and set to eval mode
    model.remove_weight_norm()
    model = model.eval()
    if is_half == True:
        model = model.half().to(device)
    else:
        model = model.to(device)

if model_version!="v3":model=None
else:init_bigvgan()


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx=audio.abs().max()
    if(maxx>1):audio/=min(2,maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

dtype=torch.float16 if is_half == True else torch.float32
def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

from text import chinese
def get_phones_and_bert(text,language,version,final=False):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        language = language.replace("all_","")
        if language == "en":
            formattext = text
        else:
            # 因无法区别中日韩文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "zh":
            if re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"zh",version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(device)
        elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext,"yue",version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist=[]
        langlist=[]
        if language == "auto":
            for tmp in LangSegmenter.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text,language,version,final=True)

    return phones,bert.to(dtype),norm_text

from module.mel_processing import spectrogram_torch,spec_to_mel_torch
def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    spec=spectrogram_torch(y,n_fft,sampling_rate,hop_size,win_size,center)
    mel=spec_to_mel_torch(spec,n_fft,num_mels,sampling_rate,fmin,fmax)
    return mel
mel_fn_args = {
    "n_fft": 1024,
    "win_size": 1024,
    "hop_size": 256,
    "num_mels": 100,
    "sampling_rate": 24000,
    "fmin": 0,
    "fmax": None,
    "center": False
}

spec_min = -12
spec_max = 2
def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1
def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min
mel_fn=lambda x: mel_spectrogram(x, **mel_fn_args)


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

##ref_wav_path+prompt_text+prompt_language+text(单个)+text_language+top_k+top_p+temperature
# cache_tokens={}#暂未实现清理机制
cache= {}
def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut=i18n("不切"), top_k=20, top_p=0.6, temperature=0.6, ref_free = False,speed=1,if_freeze=False,inp_refs=None,sample_steps=8):
    global cache
    if ref_wav_path:pass
    else:gr.Warning(i18n('请上传参考音频'))
    if text:pass
    else:gr.Warning(i18n('请填入推理文本'))
    t = []
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    if model_version=="v3":ref_free=False#s2v3暂不支持ref_free
    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]


    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)
    text = text.strip("\n")
    # if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text
    
    print(i18n("实际输入的目标文本:"), text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    if not ref_free:
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                gr.Warning(i18n("参考音频在3~10秒范围外，请更换！"))
                raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if is_half == True:
                wav16k = wav16k.half().to(device)
                zero_wav_torch = zero_wav_torch.half().to(device)
            else:
                wav16k = wav16k.to(device)
                zero_wav_torch = zero_wav_torch.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)

    t1 = ttime()
    t.append(t1-t0)

    if (how_to_cut == i18n("凑四句一切")):
        text = cut1(text)
    elif (how_to_cut == i18n("凑50字一切")):
        text = cut2(text)
    elif (how_to_cut == i18n("按中文句号。切")):
        text = cut3(text)
    elif (how_to_cut == i18n("按英文句号.切")):
        text = cut4(text)
    elif (how_to_cut == i18n("按标点符号切")):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    print(i18n("实际输入的目标文本(切句后):"), text)
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    ###s2v3暂不支持ref_free
    if not ref_free:
        phones1,bert1,norm_text1=get_phones_and_bert(prompt_text, prompt_language, version)

    for i_text,text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        if (text[-1] not in splits): text += "。" if text_language != "en" else "."
        print(i18n("实际输入的目标文本(每句):"), text)
        phones2,bert2,norm_text2=get_phones_and_bert(text, text_language, version)
        print(i18n("前端处理后的文本(每句):"), norm_text2)
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1+phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        t2 = ttime()
        # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
        # print(cache.keys(),if_freeze)
        if(i_text in cache and if_freeze==True):pred_semantic=cache[i_text]
        else:
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=hz * max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                cache[i_text]=pred_semantic
        t3 = ttime()
        ###v3不存在以下逻辑和inp_refs
        if model_version!="v3":
            refers=[]
            if(inp_refs):
                for path in inp_refs:
                    try:
                        refer = get_spepc(hps, path.name).to(dtype).to(device)
                        refers.append(refer)
                    except:
                        traceback.print_exc()
            if(len(refers)==0):refers = [get_spepc(hps, ref_wav_path).to(dtype).to(device)]
            audio = (vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers,speed=speed).detach().cpu().numpy()[0, 0])
        else:
            refer = get_spepc(hps, ref_wav_path).to(device).to(dtype)#######这里要重采样切到32k,因为src是24k的，没有单独的32k的src，所以不能改成2个路径
            phoneme_ids0=torch.LongTensor(phones1).to(device).unsqueeze(0)
            phoneme_ids1=torch.LongTensor(phones2).to(device).unsqueeze(0)
            fea_ref,ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
            ref_audio, sr = torchaudio.load(ref_wav_path)
            ref_audio=ref_audio.to(device).float()
            if (ref_audio.shape[0] == 2):
                ref_audio = ref_audio.mean(0).unsqueeze(0)
            if sr!=24000:
                ref_audio=resample(ref_audio,sr)
            mel2 = mel_fn(ref_audio.to(dtype))
            mel2 = norm_spec(mel2)
            T_min = min(mel2.shape[2], fea_ref.shape[2])
            mel2 = mel2[:, :, :T_min]
            fea_ref = fea_ref[:, :, :T_min]
            if (T_min > 468):
                mel2 = mel2[:, :, -468:]
                fea_ref = fea_ref[:, :, -468:]
                T_min = 468
            chunk_len = 934 - T_min
            fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge)
            cfm_resss = []
            idx = 0
            while (1):
                fea_todo_chunk = fea_todo[:, :, idx:idx + chunk_len]
                if (fea_todo_chunk.shape[-1] == 0): break
                idx += chunk_len
                fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                cfm_res = vq_model.cfm.inference(fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0)
                cfm_res = cfm_res[:, :, mel2.shape[2]:]
                mel2 = cfm_res[:, :, -T_min:]
                fea_ref = fea_todo_chunk[:, :, -T_min:]
                cfm_resss.append(cfm_res)
            cmf_res = torch.cat(cfm_resss, 2)
            cmf_res = denorm_spec(cmf_res)
            if model==None:init_bigvgan()
            with torch.inference_mode():
                wav_gen = model(cmf_res)
                audio=wav_gen[0][0].cpu().detach().numpy()
            max_audio=np.abs(audio).max()#简单防止16bit爆音
            if max_audio>1:audio/=max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
        t.extend([t2 - t1,t3 - t2, t4 - t3])
        t1 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % 
           (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3]))
           )
    sr=hps.data.sampling_rate if model_version!="v3"else 24000
    yield sr, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return  "\n".join(opts)

def cut4(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip(".").split(".")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def process_text(texts):
    _text=[]
    if all(text in [None, " ", "\n",""] for text in texts):
        raise ValueError(i18n("请输入有效文本"))
    for text in texts:
        if text in  [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text


def change_choices():
    SoVITS_names, GPT_names = get_weights_names(GPT_weight_root, SoVITS_weight_root)
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


SoVITS_weight_root=["SoVITS_weights","SoVITS_weights_v2","SoVITS_weights_v3"]
GPT_weight_root=["GPT_weights","GPT_weights_v2","GPT_weights_v3"]
for path in SoVITS_weight_root+GPT_weight_root:
    os.makedirs(path,exist_ok=True)


def get_weights_names(GPT_weight_root, SoVITS_weight_root):
    SoVITS_names = [i for i in pretrained_sovits_name]
    for path in SoVITS_weight_root:
        for name in os.listdir(path):
            if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (path, name))
    GPT_names = [i for i in pretrained_gpt_name]
    for path in GPT_weight_root:
        for name in os.listdir(path):
            if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (path, name))
    return SoVITS_names, GPT_names


SoVITS_names, GPT_names = get_weights_names(GPT_weight_root, SoVITS_weight_root)

def html_center(text, label='p'):
    return f"""<div style="text-align: center; margin: 100; padding: 50;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""

def html_left(text, label='p'):
    return f"""<div style="text-align: left; margin: 0; padding: 0;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""

@torch.no_grad()
def get_code_from_ssl(ssl):
    ssl = vq_model.ssl_proj(ssl)
    quantized, codes, commit_loss, quantized_list = vq_model.quantizer(ssl)
    # print(codes.shape, codes.dtype)  # [n_q, B, T]
    return codes.transpose(0, 1)  # [B, n_q, T]

    
@torch.no_grad()
def get_code_from_wav(wav_path):
    wav16k, sr = librosa.load(wav_path, sr=16000)
    wav16k = torch.from_numpy(wav16k)
    if is_half == True:
        wav16k = wav16k.half().to(device)
    else:
        wav16k = wav16k.to(device)
    ssl_content = ssl_model.model(wav16k.unsqueeze(0))[ 
        "last_hidden_state"
    ].transpose(
        1, 2
    )  # .float()
    codes = get_code_from_ssl(ssl_content)  # [B, n_q, T]

    prompt_semantic = codes[0, 0] 
    return prompt_semantic


def vc_main(wav_path, text, language, prompt_wav, noise_scale=0.5, top_k=20, top_p=0.6, temperature=0.6, speed=1, sample_steps=8):
    """
    Voice Conversion function that supports both v2 and v3 model versions
    
    Args:
        wav_path: Path to source audio for conversion
        text: Corresponding text for phoneme extraction
        language: Language of the text
        prompt_wav: Path to target/reference voice
        noise_scale: Noise scale for v2 models
        top_k, top_p, temperature: Parameters for v3 models
        speed: Speed factor for audio playback
        sample_steps: Number of sample steps for v3 models
    
    Returns:
        Sampling rate and converted audio
    """
    # Get language format
    language = dict_language[language]
    
    # Get phones from text
    phones, word2ph, norm_text = clean_text_inf(text, language, version)
    
    # Get reference audio spectrogram
    refer = get_spepc(hps, prompt_wav).to(dtype).to(device)
    
    # Get codes from source audio
    source_codes = get_code_from_wav(wav_path)
    
    if model_version != "v3":
        # V1/V2 models voice conversion logic
        ge = vq_model.ref_enc(refer)  # [B, D, T/1]
        quantized = vq_model.quantizer.decode(source_codes[None, None])  # [B, D, T]
        
        # Interpolate if necessary for 25hz models
        if hps.model.semantic_frame_rate == "25hz":
            quantized = F.interpolate(
                quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
            )
        
        m_p, logs_p, y_mask = vq_model.enc_p(
            quantized, 
            torch.LongTensor([quantized.shape[-1]]).to(device),
            torch.LongTensor(phones).to(device).unsqueeze(0),
            torch.LongTensor([len(phones)]).to(device),
            ge
        )
        
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = vq_model.flow(z_p, y_mask, g=ge, reverse=True)
        o = vq_model.dec((z * y_mask)[:, :, :], g=ge)  # [B, D=1, T], torch.float32 (-1, 1)
        audio = o.detach().cpu().numpy()[0, 0]
        
    else:
        # V3 model voice conversion logic
        if model is None:
            init_bigvgan()
        
        # Handle 1D tensor case (the source_codes from get_code_from_wav is 1D)
        # The shape of source_codes is [T] - just the sequence length
        if source_codes.dim() == 1:  # If [T]
            # For v3 models, we need to reshape to [B, T, D]
            # We need to determine the feature dimension D
            # From the error message, we can see the tensor has shape [225]
            # This is likely just the sequence length, and we need to add feature dimension
            
            # First, reshape to [1, T, 1] - adding batch and feature dimensions
            semantic = source_codes.unsqueeze(0).unsqueeze(-1)
            
            # The feature dimension may need to be expanded to match what the model expects
            # This depends on the model architecture - let's try using the same dimension as SSL features
            if hasattr(vq_model, 'ssl_dim'):
                feature_dim = vq_model.ssl_dim
            else:
                # If we can't determine it, use a default value that seems reasonable
                # For v3 models, this is often 768 (BERT/HuBERT hidden size)
                feature_dim = 768
                
            # Expand the feature dimension to match expected size
            semantic = semantic.expand(-1, -1, feature_dim)
            
        elif source_codes.dim() == 2:  # If [T, D]
            semantic = source_codes.unsqueeze(0)  # Add batch dimension [1, T, D]
        elif source_codes.dim() == 3:  # If [B, T, D]
            semantic = source_codes
        else:
            # For any other unexpected shape
            raise ValueError(f"Unexpected source_codes shape: {source_codes.shape}")
        
        # Prepare phoneme IDs
        phoneme_ids = torch.LongTensor(phones).to(device).unsqueeze(0)
        
        # Get reference audio features and global embedding
        fea_ref, ge = vq_model.decode_encp(semantic, phoneme_ids, refer)
        
        # Load and process reference audio
        ref_audio, sr = torchaudio.load(prompt_wav)
        ref_audio = ref_audio.to(device).float()
        if ref_audio.shape[0] == 2:  # Convert stereo to mono
            ref_audio = ref_audio.mean(0).unsqueeze(0)
        if sr != 24000:
            ref_audio = resample(ref_audio, sr)
        
        # Convert to mel spectrogram and normalize
        mel2 = mel_fn(ref_audio.to(dtype))
        mel2 = norm_spec(mel2)
        
        # Adjust time dimensions
        T_min = min(mel2.shape[2], fea_ref.shape[2])
        mel2 = mel2[:, :, :T_min]
        fea_ref = fea_ref[:, :, :T_min]
        
        if T_min > 468:
            mel2 = mel2[:, :, -468:]
            fea_ref = fea_ref[:, :, -468:]
            T_min = 468
        
        # Process source audio features with phoneme conditioning
        fea_todo, ge = vq_model.decode_encp(semantic, phoneme_ids, refer, ge)
        
        # Process audio in chunks
        chunk_len = 934 - T_min
        cfm_resss = []
        idx = 0
        
        while True:
            fea_todo_chunk = fea_todo[:, :, idx:idx + chunk_len]
            if fea_todo_chunk.shape[-1] == 0:
                break
            
            idx += chunk_len
            fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
            cfm_res = vq_model.cfm.inference(
                fea, 
                torch.LongTensor([fea.size(1)]).to(fea.device), 
                mel2, 
                sample_steps, 
                inference_cfg_rate=0
            )
            
            cfm_res = cfm_res[:, :, mel2.shape[2]:]
            mel2 = cfm_res[:, :, -T_min:]
            fea_ref = fea_todo_chunk[:, :, -T_min:]
            cfm_resss.append(cfm_res)
        
        # Concatenate results and convert to audio
        cmf_res = torch.cat(cfm_resss, 2)
        cmf_res = denorm_spec(cmf_res)
        
        with torch.inference_mode():
            wav_gen = model(cmf_res)
            audio = wav_gen[0][0].cpu().detach().numpy()
    
    # Normalize audio to prevent clipping
    max_audio = np.abs(audio).max()
    if max_audio > 1:
        audio /= max_audio
    
    sr = hps.data.sampling_rate if model_version != "v3" else 24000
    return sr, (audio * 32768).astype(np.int16)

# Create and launch the standalone Gradio interface for voice conversion
def launch_vc_ui():
    with gr.Blocks(title="GPT-SoVITS Voice Conversion") as vc_app:
        gr.Markdown("# GPT-SoVITS Voice Conversion")
        gr.Markdown(f"Current Model Version: {model_version}")
        
        with gr.Row():
            with gr.Column():
                source_audio = gr.Audio(type="filepath", label="Source Audio (to be converted)")
                text_input = gr.Textbox(label="Text content of the source audio")
                language_input = gr.Dropdown(
                    choices=list(dict_language.keys()),
                    value=i18n("中文"),
                    label=i18n("语言 / Language")
                )
                target_audio = gr.Audio(type="filepath", label="Target Voice (reference)")
                
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        speed = gr.Slider(
                            minimum=0.1, maximum=5, value=1, step=0.1, 
                            label=i18n("语速 / Speed")
                        )
                    
                    if model_version != "v3":
                        noise_scale = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.5, step=0.1, 
                            label="Noise Scale (V2 models only)"
                        )
                    else:
                        noise_scale = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.5, step=0.1, 
                            label="Noise Scale (ignored for V3)",
                            visible=False
                        )
                    
                    if model_version == "v3":
                        sample_steps = gr.Slider(
                            minimum=1, maximum=30, value=8, step=1, 
                            label=i18n("采样步数 / Sample Steps")
                        )
                        top_k = gr.Slider(
                            minimum=1, maximum=100, value=20, step=1, 
                            label=i18n("Top K")
                        )
                        top_p = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.6, step=0.1, 
                            label=i18n("Top P")
                        )
                        temperature = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.6, step=0.1, 
                            label=i18n("Temperature")
                        )
                    else:
                        sample_steps = gr.Slider(
                            minimum=1, maximum=30, value=8, step=1, 
                            label=i18n("采样步数 / Sample Steps"),
                            visible=False
                        )
                        top_k = gr.Slider(
                            minimum=1, maximum=100, value=20, step=1, 
                            label=i18n("Top K"),
                            visible=False
                        )
                        top_p = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.6, step=0.1, 
                            label=i18n("Top P"),
                            visible=False
                        )
                        temperature = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.6, step=0.1, 
                            label=i18n("Temperature"),
                            visible=False
                        )
                
                go_btn = gr.Button(i18n("开始转换 / Start Conversion"), variant="primary")
            
            with gr.Column():
                output_audio = gr.Audio(label=i18n("转换后的声音 / Converted Audio"))
                status_output = gr.Markdown("Ready")
        
        def process_vc(source_path, text, lang, target_path, noise, k, p, temp, spd, steps):
            try:
                if not source_path:
                    return None, "Error: Source audio is required"
                if not target_path:
                    return None, "Error: Target audio is required"
                if not text:
                    return None, "Error: Text content is required"
                
                return vc_main(
                    source_path, text, lang, target_path, 
                    noise_scale=noise, 
                    top_k=k, 
                    top_p=p, 
                    temperature=temp, 
                    speed=spd, 
                    sample_steps=steps
                ), "Conversion completed successfully"
            except Exception as e:
                import traceback
                return None, f"Error: {str(e)}\n{traceback.format_exc()}"
        
        go_btn.click(
            fn=process_vc,
            inputs=[
                source_audio, text_input, language_input, target_audio, 
                noise_scale, top_k, top_p, temperature, speed, sample_steps
            ],
            outputs=[output_audio, status_output]
        )
    
    # Launch the app with the infer_ttswebui port + 1 to avoid conflicts
    vc_app.launch(
        share=True,
    )

if __name__ == "__main__":
    print(f"Launching Voice Conversion UI with model version: {model_version}")
    launch_vc_ui()