import logging as log
import deepspeech
import  numpy as np
from halo import Halo

class DeepSpeech():
    def __init__(self, model_file, scorer_file, vad_audio):
        log.info("DeepSpeech model: {}".format(model_file))
        self.model = deepspeech.Model(model_file)

        log.info("DeepSpeech scorer: {}".format(scorer_file))
        self.model.enableExternalScorer(scorer_file)
        self.spinner = Halo(spinner='line')
        self.vad_audio = vad_audio
        self.frames = vad_audio.vad_collector()

    def inference(self):
        # Stream from microphone to DeepSpeech using VAD
        self.vad_audio.clear()
        log.info('Started capturing audio')
        stream_context = self.model.createStream()
        for frame in self.frames:
            if frame is not None:
                self.spinner.start()
                stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            else:
                self.spinner.stop()
                text = stream_context.finishStream()
                return text


