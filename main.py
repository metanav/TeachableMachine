import sys
import time
import logging as log
from subprocess import Popen, PIPE
from ScanText import ScanText
from VADAudio import VADAudio
from DeepSpeech import DeepSpeech
from BERT import BERT

# Text to speech using flite (a lite version of festival)
def tts(text):
    cmd = 'echo "{}" | festival  --tts'.format(text)
    p   = Popen(cmd, stderr=PIPE, stdout=PIPE, shell=True)
    output, errors = p.communicate()
    if p.returncode or errors:
        log.error('tts failed')


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)
    device       = 'MYRIAD'
    vocab_file   = 'models/vocab.txt'
    model_xml    = 'models/bert-small-uncased-whole-word-masking-squad-0001.xml'
    model_bin    = 'models/bert-small-uncased-whole-word-masking-squad-0001.bin'
    input_names  = ['input_ids', 'attention_mask', 'token_type_ids']
    output_names = ['output_s', 'output_e']

    bert = BERT(
        model_xml=model_xml, 
        model_bin=model_bin, 
        input_names=input_names,
        output_names=output_names,
        vocab_file=vocab_file, 
        device='MYRIAD')

    # Start audio with VAD
    vad_audio = VADAudio(aggressiveness=3, input_rate=16000)

    # Load DeepSpeech model
    ds_model_file  = 'models/deepspeech-0.8.2-models.tflite'
    ds_scorer_file = 'models/deepspeech-0.8.2-models.scorer'

    deep_speech = DeepSpeech(
        model_file=ds_model_file, 
        scorer_file=ds_scorer_file,
        vad_audio=vad_audio
    )

    scan_text = ScanText(btn_pin=17, callback=bert.set_context)
    log.info('Context is not set. Please press the button and capture the text using the camera')

    while True:
        if bert.context is not None:
            tts('Please ask a question.')
            question = deep_speech.inference()
            log.info("Question: {}".format(question))

            if question:
                answer = bert.inference(question)
                if answer:
                    tts(answer)


if __name__ == '__main__':
    sys.exit(main() or 0)
