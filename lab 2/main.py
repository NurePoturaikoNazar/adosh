import os
from glob import glob
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


EXPECTED_FRAMERATE = 22050
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 1  # (8-bit)


def ensure_outputs_dir(base_dir):
    out = os.path.join(base_dir, 'outputs')
    os.makedirs(out, exist_ok=True)
    return out


def load_wav_lab2(path):
    if not os.path.isfile(path):
        return None, None
    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype == np.uint8:
        return fs, data
    if data.dtype == np.int16:
        data16 = data.astype(np.int32)
        data_uint8 = (((data16 + 32768) * 255) // 65535).astype(np.uint8)
        return fs, data_uint8
    data_f = data.astype(np.float64)
    mn, mx = data_f.min(), data_f.max()
    if mx == mn:
        return fs, np.full(data_f.shape, 128, dtype=np.uint8)
    data_u8 = ((data_f - mn) / (mx - mn) * 255).astype(np.uint8)
    return fs, data_u8


def convert_analog_to_pulse(data_u8, k=0.8):
    a = np.abs(data_u8.astype(np.int32) - 128)
    amax = int(a.max())
    amin = int(a.min())
    p = k * (amax - amin)
    if p < 1:
        p = 5.0
    s = 0.0
    b = np.zeros(len(a), dtype=np.uint8)
    for i, ai in enumerate(a):
        s += float(ai)
        if s >= p:
            b[i] = 1
            s -= p
        else:
            b[i] = 0
    return b, p


def write_pulse_wav(path, b, fs, one_value=50, zero_value=128):
    out = np.where(b == 1, one_value, zero_value).astype(np.uint8)
    wavfile.write(path, fs, out)


def plot_input_output(save_path, input_arr, output_arr, fs, title=''):
    t = np.arange(len(input_arr)) / fs
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(t, input_arr, linewidth=0.5)
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title(f'{title} - Input')
    axs[1].plot(t, output_arr, linewidth=0.5)
    axs[1].set_ylabel('Pulse')
    axs[1].set_xlabel('Time, s')
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def compute_v1_v2_from_pulse(b):
    n = b.size
    half = n // 2
    first = b[:half]
    second = b[half:]
    v1 = first.sum() / max(1, first.size)
    v2 = second.sum() / max(1, second.size)
    return float(v1), float(v2)


def find_speech_file(base_dir):
    sounds_dir = os.path.join(base_dir, 'sounds lab 2')
    candidates = []
    if os.path.isdir(sounds_dir):
        candidates.append(os.path.join(sounds_dir, 'speech.wav'))
        candidates.extend(sorted(glob(os.path.join(sounds_dir, 'lab_2_input_*.wav'))))
    candidates.append(os.path.join(base_dir, 'speech.wav'))
    candidates.extend(sorted(glob(os.path.join(base_dir, 'lab_2_input_*.wav'))))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def find_vowel_sources(base_dir):
    sounds_dir = os.path.join(base_dir, 'sounds lab 2')
    if os.path.isdir(sounds_dir):
        a = os.path.join(sounds_dir, 'a.wav')
        u = os.path.join(sounds_dir, 'u.wav')
        i = os.path.join(sounds_dir, 'i.wav')
        if all(os.path.isfile(p) for p in (a, u, i)):
            return [(a, 'a'), (u, 'u'), (i, 'i')]
    lab1_dir = os.path.join(base_dir, '..', 'lab 1 complete', 'sounds lab 1')
    lab1_dir = os.path.normpath(lab1_dir)
    if os.path.isdir(lab1_dir):
        combined = sorted(glob(os.path.join(lab1_dir, 'my_a-u-i_sound*.wav')))
        if combined:
            return [(combined[0], 'combined')]
    return []


def split_combined_to_three(fs, data_u8):
    frame_len = max(512, int(0.02 * fs))
    hop = frame_len // 4
    n_frames = max(1, (len(data_u8) - frame_len) // hop + 1)
    energy = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        s = i * hop
        frame = data_u8[s:s+frame_len]
        energy[i] = np.sum((frame.astype(np.int32) - 128) ** 2)

    window = np.ones(5) / 5.0
    energy_s = np.convolve(energy, window, mode='same')

    thr = max(energy_s) * 0.25
    voiced = energy_s > thr
    regions = []
    i = 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            regions.append((i, j))
            i = j
        else:
            i += 1

    centers = []
    if len(regions) >= 3:
        reg_energies = [(np.sum(energy_s[s:e]), s, e) for (s, e) in regions]
        reg_energies.sort(reverse=True)
        chosen = reg_energies[:3]
        for _, s, e in chosen:
            center_frame = (s + e) // 2
            centers.append(center_frame * hop + frame_len // 2)
    else:
        peaks, _ = find_peaks(energy_s, distance=int(0.1 * fs / hop))
        if len(peaks) >= 3:
            peaks = np.sort(peaks)[-3:]
        else:
            peaks = np.argsort(energy_s)[-3:]
        for pk in np.sort(peaks):
            centers.append(int(pk * hop + frame_len // 2))

    seg_len = int(0.45 * fs)
    segments = []
    for c in centers:
        start = max(0, int(c - seg_len // 2))
        end = min(len(data_u8), int(c + seg_len // 2))
        segments.append(data_u8[start:end])

    while len(segments) < 3:
        segments.append(np.array([], dtype=np.uint8))

    return segments


def main():
    base_dir = os.path.dirname(__file__)
    outputs = ensure_outputs_dir(base_dir)

    speech_path = find_speech_file(base_dir)
    if speech_path is None:
        print("--- 1. Обробка файлу мови: ---")
        print("Помилка: файл мови не знайдено.")
    else:
        print(f"--- 1. Обробка файлу мови: {os.path.basename(speech_path)} ---")
        fs, data = load_wav_lab2(speech_path)
        if fs is None:
            print(f"Помилка при читанні {speech_path}")
        else:
            if fs != EXPECTED_FRAMERATE:
                print(f"Попередження: частота дискретизації {fs} Hz (очікується {EXPECTED_FRAMERATE} Hz)")
            pulse, p = convert_analog_to_pulse(data, k=0.8)
            out_wav = os.path.join(outputs, 'speech_if.wav')
            write_pulse_wav(out_wav, pulse, fs, one_value=50)
            out_png = os.path.join(outputs, 'speech_analysis.png')
            write_sig = np.where(pulse == 1, 200, 128).astype(np.uint8)
            plot_input_output(out_png, data, write_sig, fs, title='speech')
            print(f"Processed speech -> {out_wav}, {out_png}")

    print("\n--- 2. Побудова трикутника (a, u, i) ---")
    vowel_sources = find_vowel_sources(base_dir)
    vowel_segments = {}
    if not vowel_sources:
        print("Не знайдено файлів голосних.")
    else:
        if vowel_sources[0][1] == 'combined':
            combined_path = vowel_sources[0][0]
            print(f"Знайдено комбінований файл: {os.path.basename(combined_path)}. Спроба автоматичного розділення...")
            fs_c, data_c = load_wav_lab2(combined_path)
            if fs_c is None:
                print(f"Помилка при читанні {combined_path}")
            else:
                segs = split_combined_to_three(fs_c, data_c)
                labels = ['a', 'u', 'i']
                for lab, seg in zip(labels, segs):
                    if seg.size == 0:
                        print(f"Не вдалося виділити сегмент для {lab}")
                        continue
                    pulse, p = convert_analog_to_pulse(seg, k=0.8)
                    out_wav = os.path.join(outputs, f'{lab}_if.wav')
                    write_pulse_wav(out_wav, pulse, fs_c, one_value=50)
                    out_png = os.path.join(outputs, f'{lab}_analysis.png')
                    write_sig = np.where(pulse == 1, 200, 128).astype(np.uint8)
                    plot_input_output(out_png, seg, write_sig, fs_c, title=lab)
                    v1, v2 = compute_v1_v2_from_pulse(pulse)
                    vowel_segments[lab] = (v1, v2)
                    print(f"Processed {lab}: v1={v1:.4f}, v2={v2:.4f}")
        else:
            for path, lab in vowel_sources:
                print(f"Обробка {os.path.basename(path)} as {lab}")
                fs_v, data_v = load_wav_lab2(path)
                if fs_v is None:
                    print(f"Помилка при читанні {path}")
                    continue
                pulse, p = convert_analog_to_pulse(data_v, k=0.8)
                out_wav = os.path.join(outputs, f'{lab}_if.wav')
                write_pulse_wav(out_wav, pulse, fs_v, one_value=50)
                out_png = os.path.join(outputs, f'{lab}_analysis.png')
                write_sig = np.where(pulse == 1, 200, 128).astype(np.uint8)
                plot_input_output(out_png, data_v, write_sig, fs_v, title=lab)
                v1, v2 = compute_v1_v2_from_pulse(pulse)
                vowel_segments[lab] = (v1, v2)
                print(f"Processed {lab}: v1={v1:.4f}, v2={v2:.4f}")

    if vowel_segments:
        xs = []
        ys = []
        labs = []
        for lab in ('a', 'u', 'i'):
            if lab in vowel_segments:
                v1, v2 = vowel_segments[lab]
                xs.append(v1)
                ys.append(v2)
                labs.append(lab)
        if xs:
            plt.figure(figsize=(6, 6))
            plt.scatter(xs, ys)
            for i, lab in enumerate(labs):
                plt.text(xs[i], ys[i], lab)
            if len(xs) > 1:
                plt.plot(xs + [xs[0]], ys + [ys[0]])
            plt.xlabel('v1 (density first half)')
            plt.ylabel('v2 (density second half)')
            plt.title('Triangle of vowels')
            plt.grid(True)
            tri_png = os.path.join(outputs, 'vowel_triangle.png')
            plt.savefig(tri_png)
            plt.close()
            print(f"Saved triangle: {tri_png}")


if __name__ == '__main__':
    main()
