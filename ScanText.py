import logging as log
import time
import io
import pytesseract
import RPi.GPIO as GPIO
from picamera.array import PiRGBArray
from picamera import PiCamera
from Pixels import Pixels

class ScanText():
    def __init__(self, btn_pin, callback):
        self.callback = callback
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(btn_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(btn_pin, GPIO.FALLING, callback=self.capture, bouncetime=1000)

    def capture(self, channel):
        # Control LEDs
        pixels = Pixels()
        pixels.wakeup()
        pixels.listen()
        i = 0
        image = None
        with PiCamera() as camera:
            camera.resolution = (1280, 720)
            camera.rotation = 180
            with PiRGBArray(camera, size=(640, 360)) as output:
                for frame in camera.capture_continuous(output, format="rgb", resize=(640, 360)):
                    # Skip initial frames
                    output.truncate(0)
                    i = i + 1
                    if i <= 5:
                        time.sleep(1)
                        continue

                    image = frame.array
                    log.info('Captured image.')
                    pixels.off()
                    time.sleep(1)
                    break

        context = pytesseract.image_to_string(image)
        log.info('Converted to text.')

        if len(context) > 20:
            self.callback(context)
        else:
            log.info('Not adequate context, length = {}. Please try again.'.format(len(context)))
       
