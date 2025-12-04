import music21
from music21 import pitch
import pandas as pd

def musicxml_to_df(musicxml_path, 
                   ms_unit = 20, 
                   default_bpm = 120, 
                   force_default_bpm=False,
                   inject_rest = True):
    try:
        # 1. MusicXMLファイルの読み込み
        score = music21.converter.parse(musicxml_path)
    except Exception as e:
        print(f"エラー: MusicXMLファイルの読み込み中に問題が発生しました: {e}")
        return

    # 2. 単旋律（最初のPart）を抽出
    # 複雑なスコアの場合、最初のPartのみを処理するのが一般的です。
    part = score.parts[0] if score.parts else score

    # 3. 音符/休符のデータを抽出・計算
    notes = []
    
    # score.flat.notesAndRestsは、Part内のすべての音符と休符を、階層構造を無視して順序通りに抽出します。
    for element in part.flatten().notesAndRests:
        notes.append(element)
    
    # 最後に全休符があったら削除する
    final = 0
    for i in range(len(notes)-1,-1,-1):
        if notes[i].fullName != "Whole Rest":
            final = i
            break
    notes = notes[0:(final+1)]
    #最初と最後に休符を挿入
    if inject_rest:
        if notes[0].name != "rest":
            notes = [music21.note.Rest()]+notes
        if notes[len(notes)-1].name != "rest":
            notes.append(music21.note.Rest())

    # BPM（テンポ）情報を取得
    # 一般に、music21は`MetronomeMark`オブジェクトからテンポ情報を取得しますが、
    # 楽譜全体または音符の直前にある最初のテンポを使用します。
    
    # 楽譜全体からテンポを探す
    tempo = score.flatten().getElementsByClass(music21.tempo.MetronomeMark).first()
    
    # 見つからない場合は、Part内から探す
    if not tempo:
        tempo = part.flatten().getElementsByClass(music21.tempo.MetronomeMark).first()
    
    # テンポが見つからなかった場合のデフォルトBPM
    bpm = default_bpm
    if not force_default_bpm and tempo and tempo.number is not None:
        bpm = tempo.number
    print(f"bpm={bpm}")
    data = []
    pos_start = 0 # 音符開始フレーム
    pos_end = 0   # 音符終了フレーム
    for i in range(len(notes)):
        element = notes[i]
        # 音名または休符を取得
        if isinstance(element, music21.note.Note):
            # 音名とオクターブを組み合わせた文字列（例: "C3", "G5"）
            pitch_name = element.pitch.nameWithOctave
        elif isinstance(element, music21.note.Rest):
            # 休符は "r"
            pitch_name = "r"
        else:
            # その他の要素（コードなど）はスキップ
            continue
        duration_obj = element.duration
        
        # music21のDurationオブジェクトのquarterLength（四分音符の長さ）を取得
        quarter_length = duration_obj.quarterLength
        # 4. 絶対的な長さをミリ秒単位で計算
        # 四分音符の長さ（秒） = 60 / BPM
        # 音符の絶対的な長さ（秒） = quarter_length * (60 / BPM)
        # 音符の絶対的な長さ（ms） = quarter_length * (60 / BPM) * 1000
        
        duration_ms = quarter_length * (60 / bpm) * 1000
        
        # 5. 指定単位（ms_unit）で丸める
        # round()は正確な丸め（四捨五入）を行いますが、ここでは最も近い単位の倍数に丸めます。
        # 例: ms_unit=20の場合、43ms -> 40ms, 47ms -> 60ms
        rounded_duration_units = round(duration_ms / ms_unit)
        duration = int(rounded_duration_units)
        pos_end = pos_start+duration-1

        # 結果をリストに追加
        data.append([pitch_name, duration, pos_start, pos_end])
        pos_start = pos_end+1

    # 7. Pandas DataFrameを作成し、CSVに出力
    df = pd.DataFrame(data, columns=["pitch", "duration","start","end"])    
    return df

def transpose_pitch(df,trans):
    pt = df["pitch"]
    for i in range(len(pt)):
        if pt[i] == "r":
            continue
        p = pitch.Pitch(pt[i])
        df.loc[i,"pitch"] = p.transpose(trans).nameWithOctave

if __name__ == "__main__":
    # 実行するMusicXMLファイルのパスを指定してください
    musicxml_file = "sample.musicxml" 

    # 20ms単位で計算を実行
    df = musicxml_to_df(musicxml_file, ms_unit=20)
    print(df)