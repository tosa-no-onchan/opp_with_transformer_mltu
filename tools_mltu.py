'''
opp_with_transformer_mltu/tools_mltu.py
'''

import os

from glob import glob
import tensorflow as tf

import numpy as np
from mltu.preprocessors import WavReader
from mltu.tensorflow.dataProvider import DataProvider
from mltu.transformers import LabelIndexer, LabelPadding, SpectrogramPadding
from configs import ModelConfigs

import cv2


"""
1. tools
"""

# https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder
def latest_checkpoint(checkpoint_dir):
    #print('checkpoint_dir:',checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        return None
    #list_of_files = glob.glob('/path/to/folder/*') # * means all if need specific format then *.csv
    list_of_files = glob(checkpoint_dir+'/*weights.h5') # * means all if need specific format then *.csv
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def latest_checkno(latest):
        #print('latest:',latest)
        ret = latest.split("/")[-1]
        ret = ret.split(".")[0]
        return int(ret.split("-")[-1])


"""
2. original dataset access
"""

def get_data(wavs, id_to_text, maxlen=50):
    """returns mapping of audio paths and transcription texts"""
    data = []
    for w in wavs:
        w=w.replace('\\', '/')
        #print("w:",w)
        id = w.split("/")[-1].split(".")[0]
        #print("id",id)
        if len(id_to_text[id]) < maxlen:
            data.append({"audio": w, "text": id_to_text[id]})
    return data

"""
## Preprocess the dataset
"""
class VectorizeChar:
    def __init__(self, max_len=52):
        #self.vocab = (
        #    ["-", "#", "<", ">"]
        #    + [chr(i + 96) for i in range(1, 27)]
        #    + [" ", ".", ",", "?"]
        #)
        #  0 - 48 : y range  cnter:24
        #  49 -"B" : obstract
        #  50 .  : not use
        #  51 ?  : none
        self.vocab=(["_0","_1","_2","_3","_4","_5","_6","_7","_8","_9"]
                    + [str(i) for i in range(10,49)]
                    # "10","11","12","13","14","15","16","17","18","19",
                    # "20","21","22","23","24","25","26","27","28","29",
                    # "30","31","32","33","34","35","36","37","38","39",
                    # "40","41","42","43","44","45","46","47","48","49",
                    +["B",".","?"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        print("VectorizeChar::__call__()")
        print(" text:",text)
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab

#def create_text_ds(data):
def create_text_ds(data,vectorizer):
    texts = [_["text"] for _ in data]
    text_ds = [vectorizer(t) for t in texts]
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    return text_ds

def path_to_audio(path):
    # spectrogram using stft
    # https://work-in-progress.hatenablog.com/entry/2020/02/26/214211

    if False:
        frame_length=200
        frame_step=80
        fft_length=256
    else:
        # https://keras.io/examples/audio/ctc_asr/ values
        # An integer scalar Tensor. The window length in samples.
        frame_length = 256
        # An integer scalar Tensor. The number of samples to step.
        frame_step = 160
        # An integer scalar Tensor. The size of the FFT to apply.
        # If not provided, uses the smallest power of 2 enclosing frame_length.
        fft_length = 384    

    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)
    audio = tf.squeeze(audio, axis=-1)
    #stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    stfts = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = (x - means) / stddevs
    audio_len = tf.shape(x)[0]
    # padding to 10 seconds
    pad_len = 2754
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    return x

def create_audio_ds(data):
    flist = [_["audio"] for _ in data]
    audio_ds = tf.data.Dataset.from_tensor_slices(flist)
    audio_ds = audio_ds.map(path_to_audio, num_parallel_calls=tf.data.AUTOTUNE)
    return audio_ds

'''
def create_tf_dataset(data,bs)
    bs: batch size
'''
#def create_tf_dataset(data, bs=4):
def create_tf_dataset(data, vectorizer, bs=4):
    audio_ds = create_audio_ds(data)
    #text_ds = create_text_ds(data)
    text_ds = create_text_ds(data,vectorizer)
    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    ds = ds.map(lambda x, y: {"source": x, "target": y})
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

"""
3. mltu class for mine 
"""
if True:
    import typing
    #from mltu.transformers import Transformer as TransformerX
    import mltu.transformers as trp
    class LabelIndexer_my(trp.Transformer):
        """Convert label to index by vocab
        
        Attributes:
            vocab (typing.List[str]): List of characters in vocab
            vocab : vectorizer.char_index
        """
        def __init__(
            self, 
            vocab: typing.List[str], char_index
            ) -> None:
            self.vocab = vocab
            self.char_index = char_index

        #def __call_org__(self, data: np.ndarray, label: np.ndarray):
        #    #print('LabelIndexer_my():#7')
        #    return data, np.array([self.vocab.index(l) for l in label if l in self.vocab])

        def __call_org__(self, data: np.ndarray, label: np.ndarray):
            print('LabelIndexer_my():#7')
            print(" label:",label)
            text="<"+label+">"
            ss=[]
            for l in text:
                if l in self.char_index:
                    n = self.char_index[l]
                    ss.append(n)
            return data, np.array(ss)

        def __call__(self, data: np.ndarray, label: np.ndarray):
            #print('LabelIndexer_my():#7')
            #print(" label:",label)
            ls = label.split(" ")
            #text="<"+label+">"
            ss=[]
            for l in ls:
                i = int(l)
                #i=l
                if i == -1:
                    i=49        # 'B'
                ss.append(i)

            return data, np.array(ss)

        def __call_test__(self, data: np.ndarray, label: np.ndarray):
            ss=[]
            #print('LabelIndexer_my():#7')
            for l in label:
                if l in self.char_index:
                    n = self.char_index[l]
                    ss.append(n)
            return data, np.array(ss)
            #return {"source":data, "target":np.array(ss)}
            #return data, np.array([self.vocab.index(l) for l in label if l in self.vocab])

    """
    入力画像データの padding を行う。
    """
    class SpectrogramPadding_my(trp.Transformer):
        """Pad spectrogram to max_spectrogram_length
        
        Attributes:
            max_spectrogram_length (int): Maximum length of spectrogram
            padding_value (int): Value to pad
        """
        def __init__(
            self, 
            max_spectrogram_length: int, 
            padding_value: int,
            append: bool = True
            ) -> None:
            self.max_spectrogram_length = max_spectrogram_length
            self.padding_value = padding_value
            self.append=append

        def __call__(self, spectrogram: np.ndarray, label: np.ndarray):
            #print("SpectrogramPadding_my::__call__()")
            #print('#1 spectrogram.shape:',spectrogram.shape)
            # spectrogram.shape: (1032, 193)  <-- sprctgram
            # spectrogram.shape: (120, 130) <-- opp image data
            if self.append==False:
                padded_spectrogram = np.pad(spectrogram, 
                    ((self.max_spectrogram_length - spectrogram.shape[0], 0),(0,0)),mode="constant",constant_values=self.padding_value)
            else:
                l,h =spectrogram.shape
                lng = self.max_spectrogram_length - l
                if lng > 0:
                    a = np.full((lng,h),self.padding_value)
                    padded_spectrogram = np.append(spectrogram, a, axis=0)
                else:
                    padded_spectrogram = spectrogram

            # ノーマライズしたいが、できない!!
            #print("padded_spectrogram.dtype:",padded_spectrogram.dtype)
            # padded_spectrogram.dtype: int64
            #print('#2 padded_spectrogram.shape:',padded_spectrogram.shape)

            # ノーマライズしてみる!!
            #padded_spectrogram_float = padded_spectrogram.astype(np.float32)
            #padded_spectrogram_float = padded_spectrogram.astype(np.float64)
            #padded_spectrogram_float = padded_spectrogram_float/255.0
            #return padded_spectrogram_float, label

            return padded_spectrogram, label

    #---------------------------
    #  DataProvider for asr
    #
    # ここが、データのMain 処理
    # 最初にコールされる。
    #---------------------------
    class DataProviderAsr(DataProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
        def __getitem__(self, index: int):
            #print('DataProviderAsr():#1')
            """ Returns a batch of data by batch index"""
            dataset_batch = self.get_batch_annotations(index)
        
            # First read and preprocess the batch data
            batch_data, batch_annotations = [], []
            for index, batch in enumerate(dataset_batch):

                data, annotation = self.process_data(batch)
                #print(" annotation:",annotation)
                # 0 - 48 : class part
                # -1  -> 49
                # 未使用 -> 50
                #print(" len(annotation):",len(annotation))
                # ここが、重要!!
                # configs.max_text_length ?
                #  len(annotation): 200

                if data is None or annotation is None:
                    self.logger.warning("Data or annotation is None, skipping.")
                    continue

                batch_data.append(data)
                batch_annotations.append(annotation)

            so=np.array(batch_data)
            #print('so.shape:',so.shape)
            # so.shape: (8, 1392, 193)
            sa=np.array(batch_annotations)
            #print('sa.shape:',sa.shape)
            # sa.shape: (8, 189)
            
            #r=[so, sa]
            #print('type',type(r))
            #print('np.shape(r[0])',np.shape(r[0]))
            #print('np.shape(r[1])',np.shape(r[1]))
            #return [r]
            # change by nishi 2024.10.1
            #print('DataProviderAsr():#99')
            return (so, sa)


"""
 opp data Reader
"""
class OppReader:
    """Read wav file with librosa and return audio and label
    
    Attributes:
        frame_length (int): Length of the frames in samples.
        frame_step (int): Step size between frames in samples.
        fft_length (int): Number of FFT components.
    """

    def __init__(
            self,
            frame_length: int = 256,
            frame_step: int = 160,
            fft_length: int = 384,
            opp_path: str = "/home/nishi/colcon_ws/src/turtlebot3_navi_my/ml_data/image",
            #input_shape=[1392, 193],
            input_shape=[800, 122],
            *args, **kwargs
    ) -> None:
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.opp_path = opp_path
        self.input_shape = input_shape

        #matplotlib.interactive(False)
        # import librosa using importlib
        #import_librosa(self)
        
    @staticmethod
    def get_spectrogram(wav_path: str, frame_length: int, frame_step: int, fft_length: int) -> np.ndarray:
        """Compute the spectrogram of a WAV file

        Args:
            wav_path (str): Path to the WAV file.
            frame_length (int): Length of the frames in samples.
            frame_step (int): Step size between frames in samples.
            fft_length (int): Number of FFT components.

        Returns:
            np.ndarray: Spectrogram of the WAV file.
        """
        if False:
            import_librosa(WavReader)

            # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
            audio, orig_sr = WavReader.librosa.load(wav_path) 

            # Compute the Short Time Fourier Transform (STFT) of the audio data and store it in the variable 'spectrogram'
            # The STFT is computed with a hop length of 'frame_step' samples, a window length of 'frame_length' samples, and 'fft_length' FFT components.
            # The resulting spectrogram is also transposed for convenience
            spectrogram = WavReader.librosa.stft(audio, hop_length=frame_step, win_length=frame_length, n_fft=fft_length).T

            # Take the absolute value of the spectrogram to obtain the magnitude spectrum
            spectrogram = np.abs(spectrogram)

            # Take the square root of the magnitude spectrum to obtain the log spectrogram
            spectrogram = np.power(spectrogram, 0.5)

            # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation.
            # A small value of 1e-10 is added to the denominator to prevent division by zero.
            spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-10)

            return spectrogram
        return 0
    
    def get_opp_data(self,img_path: str) -> np.ndarray:
        #print("OppReader:get_opp_data(): #1")

        im_gray = cv2.imread(self.opp_path+'/'+img_path, cv2.IMREAD_GRAYSCALE)
        #print(" type(im_gray):",type(im_gray))
        # type(im_gray): <class 'numpy.ndarray'>
        #print(" im_gray.shape:",im_gray.shape)
        # im_gray.shape: (120, 134)
        #print(" im_gray.shape:",im_gray.shape)

        #print("im_gray.dtype:",im_gray.dtype)
        # im_gray.dtype: uint8

        # old img size (120,cols) -> new img size (122,cols)
        if(im_gray.shape[0] <  self.input_shape[1]):
            rowx=self.input_shape[1] - im_gray.shape[0]
            a = np.full((rowx, im_gray.shape[1]),255,dtype=np.uint8)
            im_gray=np.append(im_gray, a, axis=0)
            #print(" im_gray.shape:",im_gray.shape)

        #  rows,cols -> cols,rows に変換必要か
        # (120,134)  -> (134,120)
        #  a_2d_T = a_2d.T
        return im_gray.T

    @staticmethod
    def plot_raw_audio(wav_path: str, title: str = None, sr: int = 16000) -> None:
        """Plot the raw audio of a WAV file

        Args:
            wav_path (str): Path to the WAV file.
            sr (int, optional): Sample rate of the WAV file. Defaults to 16000.
            title (str, optional): Title
        """
        if False:
            #import_librosa(WavReader)
            # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
            audio, orig_sr = WavReader.librosa.load(wav_path, sr=sr)

            duration = len(audio) / orig_sr

            time = np.linspace(0, duration, num=len(audio))

            #plt.figure(figsize=(15, 5))
            #plt.plot(time, audio)
            #plt.title(title) if title else plt.title("Audio Plot")
            #plt.ylabel("signal wave")
            #plt.xlabel("time (s)")
            #plt.tight_layout()
            #plt.show()

    @staticmethod
    def plot_spectrogram(spectrogram: np.ndarray, title:str = "", transpose: bool = True, invert: bool = True) -> None:
        """Plot the spectrogram of a WAV file

        Args:
            spectrogram (np.ndarray): Spectrogram of the WAV file.
            title (str, optional): Title of the plot. Defaults to None.
            transpose (bool, optional): Transpose the spectrogram. Defaults to True.
            invert (bool, optional): Invert the spectrogram. Defaults to True.
        """
        if transpose:
            spectrogram = spectrogram.T
        
        if invert:
            spectrogram = spectrogram[::-1]

        #plt.figure(figsize=(15, 5))
        #plt.imshow(spectrogram, aspect="auto", origin="lower")
        #plt.title(f"Spectrogram: {title}")
        #plt.xlabel("Time")
        #plt.ylabel("Frequency")
        #plt.colorbar()
        #plt.tight_layout()
        #plt.show()

    def __call__(self, img_path: str, label: typing.Any):
        """
        Extract the spectrogram and label of a WAV file.

        Args:
            audio_path (str): Path to the WAV file.
            label (typing.Any): Label of the WAV file.

        Returns:
            Tuple[np.ndarray, typing.Any]: Spectrogram of the WAV file and its label.
        """
        #print("OppReader:__call__()")
        #print(" img_path:",img_path)
        #print(" label:",label)

        #return self.get_spectrogram(audio_path, self.frame_length, self.frame_step, self.fft_length), label
        return self.get_opp_data(img_path), label