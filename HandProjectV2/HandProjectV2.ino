#define CAMERA_MODEL_AI_THINKER
#include <WiFi.h>
#include <WebServer.h>
#include <esp_camera.h>
#include "camera_pin.h"

#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

//load model
#include "HandModel.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
 TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// Create an area of memory to use for input, output, and other TensorFlow
  // arrays. You'll need to adjust this by combiling, running, and looking
  // for errors.
  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

bool setup_camera(framesize_t frameSize) {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_GRAYSCALE;
    config.frame_size = frameSize;
    config.jpeg_quality = 12;
    config.fb_count = 1;

    // disable white balance and white balance gain
    // sensor_t * sensor = esp_camera_sensor_get();
    // sensor->set_whitebal(sensor, 0);       // 0 = disable , 1 = enable
    // sensor->set_awb_gain(sensor, 0);       // 0 = disable , 1 = enable

    return esp_camera_init(&config) == ESP_OK;
}

void setup() {
  Serial.begin(115200);
  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  // Map the model into a usable data structure
  model = tflite::GetModel(HandModel);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  if (!setup_camera(FRAMESIZE_QQVGA)) {
    Serial.println("Camera init failed");
    return;
  }
}

void handle_capture()
{
    camera_fb_t * frame = esp_camera_fb_get();
    if (!frame) {
        Serial.println("Camera capture failed");
        return;
    }

    int RESIZE_WIDTH = 28; // Di chuyển khai báo của biến
    int RESIZE_HEIGHT = 28; // Di chuyển khai báo của biến
    
    // Calculate scale factors for resizing
    int scale_x = frame->width / RESIZE_WIDTH;
    int scale_y = frame->height / RESIZE_HEIGHT;

    // Print resized pixel values
    for (int y = 0; y < RESIZE_HEIGHT; y++) {
        for (int x = 0; x < RESIZE_WIDTH; x++) {
            int orig_x = x * scale_x;
            int orig_y = y * scale_y;
            int sum = 0;

            // Calculate average of pixels in group
            for (int dy = 0; dy < scale_y; dy++) {
                for (int dx = 0; dx < scale_x; dx++) {
                    int orig_index = ((orig_y + dy) * frame->width) + (orig_x + dx);
                    sum += frame->buf[orig_index];
                }
            }

            // Calculate average pixel value
            uint8_t avg_pixel = sum / (scale_x * scale_y);
            model_input->data.f[x*y] = avg_pixel;
        }
    }

    esp_camera_fb_return(frame);
}
void loop() {
  handle_capture();

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
  }

  int predict = 0;
  int max = -999;

  // Print the output to the Serial
    for (int i = 0; i < model_output->dims->data[1]; i++) {
      float value = model_output->data.f[i];
      if (value > max){
        max = value;
        predict = i;
      }
    }
    Serial.print("Predict = ");
    Serial.print(predict);
    Serial.println("");
  
}
