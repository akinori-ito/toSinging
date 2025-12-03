import pyworld as pw
import librosa
import numpy as np
import soundfile as sf
import vuv_model as vuv
import dtw
from noteseq import noteseq
import MXML2 as mxml
import sys
import vibration

def analyze(wavfile=None,w=None,sr=None,fftspec=False):
    assert not (w is None) or not (wavfile is None), "Either w or wavfile should be specified"
    if w is None:
        w, sr = librosa.load(wavfile, sr=16000)
    else:
        assert not (sr is None), "When specifying w, sr must be specified"
    yf = np.array(w,dtype=np.float64)
    _f0, t = pw.harvest(yf,sr)
    f0 = pw.stonemask(yf,_f0, t, sr)
    ap = pw.d4c(yf,f0,t,sr)
    sp = pw.cheaptrick(yf,f0,t,sr)
    if fftspec:
        sp_world = sp
        sp = librosa.stft(w,n_fft=1024,hop_length=80).T
        sp = np.abs(sp)**2
        length,_ = sp.shape
        if length > ap.shape[0]:
            sp = sp[0:ap.shape[0],:]
        elif length < ap.shape[0]:
            nsp = np.zeros(ap.shape)
            nsp[0:sp.shape[0],:] = sp
            sp = nsp
        sp = sp/np.max(sp)*np.max(sp_world)
    return({"spectrum": sp, "aperiodicity": ap, "f0": f0, "rate": sr})

def synthesize(anl,replace_ap = True):
    np.save("z_spectrum.npy",anl["spectrum"])
    np.save("z_aperiodicity.npy",anl["aperiodicity"])
    np.save("z_f0.npy",anl["f0"])
    np.save("z_vuv.npy",anl["vuv"])
    np.save("z_orgf0.npy",anl["org_f0"])
    ap = anl["aperiodicity"]
    f0 = anl["f0"]
    if replace_ap:
        vuv = anl["vuv"]
        org_f0 = anl["org_f0"]
        voiced_ap = ap[org_f0>0].mean(axis=0)
        unvoiced_ap = ap[org_f0 == 0].mean(axis=0)

        for i in range(len(vuv)):
            if vuv[i] == 1 and org_f0[i] == 0:
                # Worldで無声と判定されたが、実際には有声
                ap[i,:] = voiced_ap
            elif vuv[i] != 1 and org_f0[i] > 0:
                # Worldで有声と判定されたが、実際には無声
                ap[i,:] = unvoiced_ap
            elif vuv[i] != 1 and org_f0[i] == 0:
                f0[i] = 0
            elif vuv[i] == 0 or f0[i] < 40:
                f0[i] = 0
    np.save("z_mod_aper.npy",ap)
    np.save("z_mod_f0.npy",f0)

    return pw.synthesize(f0,anl["spectrum"],ap,anl["rate"])

def convert_speech2sing(inputfile,musicxmlfile,outputfile,
                        modelfile,modelname,
                        bpm=None):
    print("Reading model...")
    model = vuv.VUVmodel(modelfile,modelname,3,16000) 
    print(f"Input file: {inputfile}")
    w, sr = librosa.load(inputfile,sr=16000)
    anl = analyze(w=w,sr=sr)

    print(f"MusicXML file: {musicxmlfile}")
    if bpm is None:
        src_notes = mxml.musicxml_to_df(musicxmlfile)
    else:
        src_notes = mxml.musicxml_to_df(musicxmlfile,default_bpm=bpm,force_default_bpm=True)

    notes,pitches = noteseq(src_notes)
    mapper = {0:0, 1:10, 2:2}
    print("Voiced/Unvoiced classification")
    vuv0 = model.predict(inputfile)
    vuv1 = [mapper[x] for x in vuv0]
    opt = dtw.dtw(vuv1,notes,uv_val=2)
    np.save("z_opt.npy",np.array(opt))
    N = len(pitches)
    sp = np.zeros((N*4,anl["spectrum"].shape[1]))
    ap = np.zeros((N*4,anl["aperiodicity"].shape[1]))
    f0 = np.zeros(N*4)
    vu = np.zeros(N*4)
    org_f0 = np.zeros(N*4)
    print("Calculating alignment")
    for i in range(N):
        for j in range(4):
            sp[i*4+j,:] = anl["spectrum"][opt[i][0]*4+j,:]
            ap[i*4+j,:] = anl["aperiodicity"][opt[i][0]*4+j,:]
            f0[i*4+j] = pitches[opt[i][1]]
            vu[i*4+j] = vuv0[opt[i][0]]
            org_f0[i*4+j] = anl["f0"][opt[i][0]*4+j]
    f0 = np.ascontiguousarray(vibration.add_dumping(f0))
    anl2 = {"f0":f0,"spectrum":sp, "aperiodicity":ap, "rate":sr, "vuv": vu, "org_f0": org_f0}
    print("Synthesizing singing voice")
    w2 = synthesize(anl2)
    sf.write(outputfile,w2,anl["rate"])
    print("done")

def usage():
    print("Usage: python tosinging.py -i input.wav -m musicxml [-o output.wav] [-modelfile VUV_modelfile] [-modelname modelname] [-bpm bpm]")
    exit()

### MAIN
inputfile = None
musicxmlfile = None
outputfile = "output.wav"
modelfile = "hubert_sad_20ms_model.pth"
modelname = "facebook/hubert-base-ls960"
bpm = None

i = 1
while i < len(sys.argv):
    if sys.argv[i] == "-i":
        i += 1
        inputfile = sys.argv[i]
    elif sys.argv[i] == "-m":
        i += 1
        musicxmlfile = sys.argv[i]
    elif sys.argv[i] == "-o":
        i += 1
        outputfile = sys.argv[i]
    elif sys.argv[i] == "-modelfile":
        i += 1
        modelfile = sys.argv[i]
    elif sys.argv[i] == "-modelname":
        i += 1
        modelname = sys.argv[i]
    elif sys.argv[i] == "-bpm":
        i += 1
        bpm = int(sys.argv[i])
    else:
        print("Unknown option:",sys.argv[i])
        usage()
    i += 1

if inputfile is None:
    print("No input file")
    usage()
if musicxmlfile is None:
    print("No MusicXML file")
    usage()

convert_speech2sing(inputfile,musicxmlfile,outputfile,modelfile,modelname,bpm=bpm)
