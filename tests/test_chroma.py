import numpy as np


def test_float_audio_chroma_matches_the_existing_pcm_path(tmp_path):
    """Float transformation controls must use exactly the existing chroma."""
    import wave
    from veridis_tille.chroma import extract_chroma, extract_chroma_samples
    sr = 44100
    time = np.arange(sr) / sr
    pcm = (np.sin(2*np.pi*440*time) * 12000).astype("<i2")
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, sr, 0, "NONE", "not compressed"))
        stream.writeframes(pcm.tobytes())
    old = extract_chroma(path)
    floating = extract_chroma_samples(pcm.astype(np.float32) / 32768, sr)
    np.testing.assert_array_equal(old.values, floating.values)
    assert old.duration == floating.duration
