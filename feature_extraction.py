#!/usr/bin/env python3
"""
Feature Extraction Models - Built From Scratch
1. Visual Features: Custom CNN for frame analysis
2. NLP Features: From-scratch audio-to-text and text analysis
"""

import numpy as np
import cv2
from scipy.fft import fft
from scipy.signal import stft
import wave
import struct

# ============================================================================
# VISUAL FEATURE EXTRACTION - Simple CNN from Scratch
# ============================================================================

class SimpleCNN:
    """
    Simple Convolutional Neural Network from scratch
    Used to extract visual features from video frames
    """
    
    def __init__(self):
        self.feature_dim = 512
        # Initialize simple filters for edge detection and patterns
        self.initialize_filters()
    
    def initialize_filters(self):
        """Initialize basic convolutional filters"""
        # Horizontal edge detector
        self.h_filter = np.array([[-1, -1, -1],
                                   [0, 0, 0],
                                   [1, 1, 1]])
        
        # Vertical edge detector
        self.v_filter = np.array([[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]])
        
        # Diagonal edge detector
        self.d_filter = np.array([[0, 1, 1],
                                   [-1, 0, 1],
                                   [-1, -1, 0]])
    
    def convolve2d(self, image, kernel):
        """Basic 2D convolution operation"""
        k_h, k_w = kernel.shape
        i_h, i_w = image.shape
        
        # Output dimensions
        out_h = i_h - k_h + 1
        out_w = i_w - k_w + 1
        output = np.zeros((out_h, out_w))
        
        # Perform convolution
        for i in range(out_h):
            for j in range(out_w):
                region = image[i:i+k_h, j:j+k_w]
                output[i, j] = np.sum(region * kernel)
        
        return output
    
    def extract_features(self, frame):
        """
        Extract visual features from a frame
        
        Args:
            frame: numpy array (H, W, 3) - BGR image
        
        Returns:
            feature_vector: 1D numpy array of features
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Normalize
        gray = gray.astype(np.float32) / 255.0
        
        # Apply filters
        h_edges = self.convolve2d(gray, self.h_filter)
        v_edges = self.convolve2d(gray, self.v_filter)
        d_edges = self.convolve2d(gray, self.d_filter)
        
        # Color histogram features (if color image)
        if len(frame.shape) == 3:
            hist_b = cv2.calcHist([frame], [0], None, [32], [0, 256]).flatten()
            hist_g = cv2.calcHist([frame], [1], None, [32], [0, 256]).flatten()
            hist_r = cv2.calcHist([frame], [2], None, [32], [0, 256]).flatten()
            color_features = np.concatenate([hist_b, hist_g, hist_r])
        else:
            color_features = np.zeros(96)
        
        # Texture features (using edge responses)
        texture_features = np.array([
            np.mean(h_edges),
            np.std(h_edges),
            np.mean(v_edges),
            np.std(v_edges),
            np.mean(d_edges),
            np.std(d_edges),
            np.max(h_edges),
            np.max(v_edges),
            np.max(d_edges)
        ])
        
        # Motion/intensity features
        intensity_features = np.array([
            np.mean(gray),
            np.std(gray),
            np.min(gray),
            np.max(gray)
        ])
        
        # Combine all features
        feature_vector = np.concatenate([
            color_features,
            texture_features,
            intensity_features
        ])
        
        # Pad or truncate to fixed size
        if len(feature_vector) < self.feature_dim:
            feature_vector = np.pad(feature_vector, 
                                   (0, self.feature_dim - len(feature_vector)))
        else:
            feature_vector = feature_vector[:self.feature_dim]
        
        return feature_vector


# ============================================================================
# AUDIO PROCESSING - From Scratch
# ============================================================================

class AudioProcessor:
    """
    Process audio files from scratch
    Extract features for speech-to-text
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def read_wav(self, wav_bytes):
        """Read WAV file from bytes"""
        import io
        
        # Parse WAV format manually
        wav_io = io.BytesIO(wav_bytes)
        
        # Read using wave module
        with wave.open(wav_io, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # Read audio data
            audio_data = wav_file.readframes(n_frames)
        
        # Convert to numpy array
        if sample_width == 2:  # 16-bit audio
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        else:
            audio_array = np.frombuffer(audio_data, dtype=np.uint8)
        
        # Normalize
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        return audio_array, frame_rate
    
    def extract_mfcc_features(self, audio, sample_rate):
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients)
        These are essential for speech recognition
        """
        # Parameters
        frame_size = int(0.025 * sample_rate)  # 25ms frames
        frame_stride = int(0.010 * sample_rate)  # 10ms stride
        n_mfcc = 13
        
        # Pre-emphasis filter (emphasize higher frequencies)
        emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # Frame the signal
        num_frames = int(np.ceil(len(emphasized) / frame_stride))
        pad_length = num_frames * frame_stride - len(emphasized)
        padded = np.pad(emphasized, (0, pad_length), mode='constant')
        
        # Create frames
        frames = []
        for i in range(0, len(padded) - frame_size + 1, frame_stride):
            frames.append(padded[i:i+frame_size])
        
        frames = np.array(frames)
        
        # Apply Hamming window
        frames *= np.hamming(frame_size)
        
        # FFT and power spectrum
        fft_frames = np.fft.rfft(frames, n=512)
        power_spectrum = np.abs(fft_frames) ** 2
        
        # Mel filterbank
        mel_filters = self.mel_filterbank(sample_rate, 512, n_filters=26)
        mel_energies = np.dot(power_spectrum, mel_filters.T)
        mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)
        
        # Log and DCT
        log_mel = np.log(mel_energies)
        mfcc = self.dct(log_mel, n_mfcc)
        
        return mfcc
    
    def mel_filterbank(self, sample_rate, n_fft, n_filters=26):
        """Create Mel filterbank"""
        low_freq = 0
        high_freq = sample_rate / 2
        
        # Convert to Mel scale
        low_mel = 2595 * np.log10(1 + low_freq / 700)
        high_mel = 2595 * np.log10(1 + high_freq / 700)
        
        # Equal spacing in Mel scale
        mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
        
        # Convert back to Hz
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        
        # Convert to FFT bin numbers
        bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
        
        # Create filterbank
        filterbank = np.zeros((n_filters, n_fft // 2 + 1))
        
        for i in range(1, n_filters + 1):
            left = bins[i - 1]
            center = bins[i]
            right = bins[i + 1]
            
            for j in range(left, center):
                filterbank[i - 1, j] = (j - left) / (center - left)
            
            for j in range(center, right):
                filterbank[i - 1, j] = (right - j) / (right - center)
        
        return filterbank
    
    def dct(self, x, n_coeffs):
        """Discrete Cosine Transform"""
        N = x.shape[1]
        dct_matrix = np.zeros((n_coeffs, N))
        
        for k in range(n_coeffs):
            for n in range(N):
                dct_matrix[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        
        return np.dot(x, dct_matrix.T)


# ============================================================================
# SIMPLE SPEECH-TO-TEXT - From Scratch (Basic Implementation)
# ============================================================================

class SimpleASR:
    """
    Automatic Speech Recognition - Simplified from scratch
    Uses acoustic features and basic pattern matching
    """
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        
        # Simple phoneme patterns (this is very simplified)
        # In reality, you'd train on labeled data
        self.phoneme_patterns = self._initialize_phonemes()
    
    def _initialize_phonemes(self):
        """Initialize basic phoneme patterns"""
        # This is a simplified representation
        # Real ASR would use trained HMMs or neural networks
        return {
            'silence': {'energy_threshold': 0.01},
            'vowel': {'energy_threshold': 0.1, 'frequency_range': (300, 3000)},
            'consonant': {'energy_threshold': 0.05, 'frequency_range': (2000, 8000)}
        }
    
    def transcribe(self, audio_bytes):
        """
        Convert audio to text (simplified)
        
        Returns:
            transcript: string of transcribed text
        """
        # Read audio
        audio, sample_rate = self.audio_processor.read_wav(audio_bytes)
        
        # Extract MFCC features
        mfcc = self.audio_processor.extract_mfcc_features(audio, sample_rate)
        
        # Simple energy-based segmentation
        energy = np.sum(mfcc ** 2, axis=1)
        
        # Detect speech segments (simplified)
        threshold = np.mean(energy) * 0.5
        speech_frames = energy > threshold
        
        # Count speech segments as approximate word count
        # This is very simplified - real ASR is much more complex
        segments = self._find_segments(speech_frames)
        
        # Generate simplified transcript
        # In a real system, you'd use trained models
        transcript = f"[Audio contains approximately {len(segments)} speech segments]"
        
        # Return basic info
        return {
            'transcript': transcript,
            'num_segments': len(segments),
            'duration': len(audio) / sample_rate,
            'mfcc_features': mfcc.mean(axis=0).tolist()  # Average features
        }
    
    def _find_segments(self, speech_frames):
        """Find continuous speech segments"""
        segments = []
        in_segment = False
        start = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_segment:
                start = i
                in_segment = True
            elif not is_speech and in_segment:
                segments.append((start, i))
                in_segment = False
        
        if in_segment:
            segments.append((start, len(speech_frames)))
        
        return segments


# ============================================================================
# NLP - Text Processing From Scratch
# ============================================================================

class SimpleNLP:
    """
    Basic NLP for text analysis - from scratch
    """
    
    def __init__(self):
        # Simple stop words
        self.stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 
                          'as', 'are', 'was', 'were', 'been', 'be', 'have',
                          'has', 'had', 'do', 'does', 'did', 'but', 'if',
                          'or', 'and', 'to', 'of', 'in', 'for', 'with'}
    
    def tokenize(self, text):
        """Simple word tokenization"""
        # Convert to lowercase and split
        text = text.lower()
        # Remove punctuation
        for punct in '.,!?;:':
            text = text.replace(punct, ' ')
        
        tokens = text.split()
        return tokens
    
    def remove_stopwords(self, tokens):
        """Remove common stop words"""
        return [t for t in tokens if t not in self.stop_words]
    
    def extract_keywords(self, text, top_n=10):
        """Extract keywords using simple frequency"""
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        
        # Count frequencies
        freq = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_words[:top_n]
    
    def compute_tfidf(self, documents):
        """
        Compute TF-IDF from scratch
        
        Args:
            documents: list of text strings
        
        Returns:
            tfidf_matrix: numpy array of TF-IDF scores
        """
        # Tokenize all documents
        tokenized_docs = [self.tokenize(doc) for doc in documents]
        
        # Build vocabulary
        vocab = set()
        for tokens in tokenized_docs:
            vocab.update(tokens)
        vocab = sorted(list(vocab))
        vocab_to_idx = {word: i for i, word in enumerate(vocab)}
        
        # Compute term frequencies
        tf_matrix = np.zeros((len(documents), len(vocab)))
        for doc_idx, tokens in enumerate(tokenized_docs):
            for token in tokens:
                if token in vocab_to_idx:
                    tf_matrix[doc_idx, vocab_to_idx[token]] += 1
        
        # Normalize by document length
        doc_lengths = np.sum(tf_matrix, axis=1, keepdims=True)
        doc_lengths[doc_lengths == 0] = 1  # Avoid division by zero
        tf_matrix = tf_matrix / doc_lengths
        
        # Compute inverse document frequency
        doc_count = np.sum(tf_matrix > 0, axis=0)
        idf = np.log(len(documents) / (doc_count + 1))
        
        # Compute TF-IDF
        tfidf_matrix = tf_matrix * idf
        
        return tfidf_matrix, vocab


# Test functions
if __name__ == "__main__":
    print("Feature Extraction Models - From Scratch")
    print("="*60)
    
    # Test Visual CNN
    print("\n1. Testing Visual Feature Extractor...")
    cnn = SimpleCNN()
    test_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    features = cnn.extract_features(test_frame)
    print(f"   Visual features shape: {features.shape}")
    
    # Test NLP
    print("\n2. Testing NLP Processor...")
    nlp = SimpleNLP()
    test_text = "This is a sample video about machine learning and data processing"
    keywords = nlp.extract_keywords(test_text)
    print(f"   Keywords: {keywords}")
    
    print("\n✓ All models initialized successfully!")