import pympi
import os
import torch
import whisperx
import wave
import librosa
import numpy as np
import re
from pydub import AudioSegment
from pydub.silence import split_on_silence

# ==========================================
# 最終修正版：無音の長さを100%保持する分割・同期スクリプト
# ==========================================
AUDIO_FILE_L = "/content/Speaker_L.wav"
AUDIO_FILE_R = "/content/Speaker_R.wav"
BASE_EAF_PATH = "/content/Transcription.eaf"
OUTPUT_EAF_PATH = "/content/Transcription_merged.eaf"

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
compute_type = "float16"

print("WhisperXモデルをロードしています...")
model = whisperx.load_model("medium", device, compute_type=compute_type)

def convert_to_kanji_number(text):
    def replace_num(match):
        num_str = match.group(0)
        if num_str.isdigit():
            trans_table = str.maketrans('0123456789', '〇一二三四五六七八九')
            return num_str.translate(trans_table)
        return num_str
    return re.sub(r'\d+', replace_num, text)

def apply_manual_rules(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\[laughter\]|\[笑い声\]', '<la>', text)
    text = re.sub(r'\[cough\]|\[throat clearing\]|\[咳払い\]', '<no>', text)
    text = re.sub(r'\[unintelligible\]|\[不明\]', '<?>', text)
    replacement_dict = {
        "twitter": "[ツイッター] <un>",
        "Twitter": "[ツイッター] <un>",
        "CHANEL": "[シャネル] <un>",
        "chanel": "[シャネル] <un>"
    }
    for key, val in replacement_dict.items():
        text = text.replace(key, val)
    return convert_to_kanji_number(text)

def determine_voiced_or_unvoiced(audio_path, start_sec, end_sec, sr=22050):
    try:
        duration = max(0.1, end_sec - start_sec)
        y, sr = librosa.load(audio_path, sr=sr, offset=start_sec, duration=duration)
        if len(y) == 0:
            return "laugh_unvoiced"
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        if voiced_flag is None or len(voiced_flag) == 0:
            return "laugh_unvoiced"
        valid_flags = np.nan_to_num(voiced_flag, nan=0.0)
        voiced_ratio = np.sum(valid_flags > 0) / len(valid_flags)
        return "laugh_voiced" if voiced_ratio > 0.1 else "laugh_unvoiced"
    except Exception:
        return "laugh_unvoiced"

if not os.path.exists(BASE_EAF_PATH):
    print(f"エラー: ベースとなるEAFファイルが見つかりません: {BASE_EAF_PATH}")
else:
    print("EAFファイルを読み込んでいます...")
    eaf = pympi.Elan.Eaf(BASE_EAF_PATH)

    audio_files = {"L": AUDIO_FILE_L, "R": AUDIO_FILE_R}

    for speaker, audio_path in audio_files.items():
        print(f"\n--- 話者 {speaker} の正確な無音分割・タイムライン同期処理を開始 ---")
        if not os.path.exists(audio_path):
            print(f"エラー: 音声ファイルが見つかりません: {audio_path}")
            continue

        try:
            audio_segment = AudioSegment.from_file(audio_path)
            dbfs = audio_segment.dBFS
            silence_thresh = max(-50, dbfs - 18)

            # pydubのdetect_silenceを使って「無音ではない実発話の区間（start_ms, end_ms）」を正確に取得する
            # これにより、間の長い無音部分の長さが完全に維持されます
            from pydub.silence import detect_nonsilent
            nonsilent_ranges = detect_nonsilent(
                audio_segment,
                min_silence_len=500,  # 0.5秒以上の無音を境界とする
                silence_thresh=silence_thresh,
                seek_step=10
            )

            print(f"話者 {speaker}: {len(nonsilent_ranges)} 個の有効な音声発話ブロックに分割されました。")

            trans_tier = f"{speaker}_Speaker_Transcription"
            laugh_tier = f"{speaker}_Speaker_Laugh"

            for t in [trans_tier, laugh_tier]:
                if t not in eaf.get_tier_names():
                    eaf.add_tier(t)

            for idx, (block_start_ms, block_end_ms) in enumerate(nonsilent_ranges):
                # 各ブロックを切り出して文字起こし
                chunk = audio_segment[block_start_ms:block_end_ms]
                chunk_filename = f"/content/temp_block_{speaker}_{idx}.wav"
                chunk.export(chunk_filename, format="wav")

                try:
                    audio_arr = whisperx.load_audio(chunk_filename)
                    result = model.transcribe(audio_arr, batch_size=batch_size, language="ja")
                    lang_code = result.get("language", "ja")

                    model_a, metadata = whisperx.load_align_model(language_code=lang_code, device=device)
                    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio_arr, device, return_char_alignments=False)
                    segments_data = aligned_result.get("segments", [])

                    for seg in segments_data:
                        start = seg.get("start", 0.0)
                        end = seg.get("end", 0.0)
                        raw_text = seg.get("text", "")
                        word_segs = seg.get("words", [])

                        # ブロックの絶対的な開始位置（block_start_ms）をベースに加算することで、間の無音を完全に保持
                        start_ms = block_start_ms + int(start * 1000)
                        end_ms = block_start_ms + int(end * 1000)

                        formatted_text = apply_manual_rules(raw_text)

                        # 0.2秒以上0.5秒未満の間隔に読点「、」を入れる処理
                        if word_segs and len(word_segs) > 1:
                            punctuated_words = []
                            for i in range(len(word_segs) - 1):
                                w_curr = word_segs[i]
                                w_next = word_segs[i+1]
                                c_word = w_curr.get("word", "")
                                c_end = w_curr.get("end", w_next.get("start", 0))
                                n_start = w_next.get("start", c_end)

                                punctuated_words.append(c_word)
                                if 0.2 <= (n_start - c_end) < 0.5:
                                    punctuated_words.append("、")
                            punctuated_words.append(word_segs[-1].get("word", ""))
                            formatted_text = "".join(punctuated_words)

                        has_la = "<la>" in formatted_text

                        if has_la:
                            clean_text = formatted_text.replace("<la>", "").strip()
                            if len(clean_text) > 0:
                                try:
                                    eaf.add_annotation(trans_tier, start_ms, end_ms, formatted_text)
                                except Exception:
                                    pass
                            else:
                                laugh_label = determine_voiced_or_unvoiced(audio_path, start_ms / 1000.0, end_ms / 1000.0)
                                try:
                                    eaf.add_annotation(laugh_tier, start_ms, end_ms, f"<{laugh_label}>")
                                except Exception:
                                    pass
                        else:
                            if formatted_text.strip():
                                try:
                                    eaf.add_annotation(trans_tier, start_ms, end_ms, formatted_text)
                                except Exception:
                                    pass

                except Exception as sub_e:
                    print(f"ブロック {idx} の処理中にエラー: {sub_e}")

                if os.path.exists(chunk_filename):
                    os.remove(chunk_filename)

            print(f"話者 {speaker}: 無音の長さを保持した分割登録が完了しました。")

        except Exception as e:
            print(f"話者 {speaker} の処理中にエラーが発生しました: {e}")

    eaf.to_file(OUTPUT_EAF_PATH)
    print(f"\nすべての処理が完了しました！保存先: {OUTPUT_EAF_PATH}")