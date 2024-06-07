#define CAMERA_MODEL_AI_THINKER
#include <WiFi.h>
#include <WebServer.h>
#include <esp_camera.h>
#include "camera_pin.h"

#include <TensorFlowLite_ESP32.h>
#include "HandModel.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Chỉnh sửa kích thước của vùng nhớ dành cho TensorFlow Lite
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Khai báo biến interpreter là biến toàn cục
static tflite::MicroInterpreter* interpreter_ptr;

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

    return esp_camera_init(&config) == ESP_OK;
}

void handle_captureAI() {
    static bool model_allocated = false; // Biến đánh dấu cho việc cấp phát bộ nhớ cho mô hình

    // Set up logging
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure
    const tflite::Model* model = tflite::GetModel(HandModel);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report("Model provided is schema version %d not equal to supported version %d.",
                               model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Chỉ cấp phát bộ nhớ cho mô hình nếu chưa được thực hiện trước đó
    if (!model_allocated) {
        // This pulls in all the operation implementations we need.
        static tflite::AllOpsResolver resolver;

        // Build an interpreter to run the model with
        static tflite::MicroInterpreter interpreter(
            model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

        interpreter_ptr = &interpreter;

        // Allocate memory from the tensor_arena for the model's tensors
        TfLiteStatus allocate_status = interpreter.AllocateTensors();
        if (allocate_status != kTfLiteOk) {
            error_reporter->Report("AllocateTensors() failed");
            return;
        }

        model_allocated = true; // Đánh dấu rằng bộ nhớ đã được cấp phát cho mô hình
    }

    // Get information about the input tensor
    TfLiteTensor* input = interpreter_ptr->input(0);
    if (input->dims->size != 4 || input->dims->data[0] != 1 ||
        input->dims->data[1] != 28 || input->dims->data[2] != 28 ||
        input->dims->data[3] != 1) {
        error_reporter->Report("Bad input tensor parameters in model");
        return;
    }

    camera_fb_t * frame = esp_camera_fb_get();
    if (!frame) {
        Serial.println("Camera capture failed");
        return;
    }
    esp_camera_fb_return(frame);

    frame = esp_camera_fb_get();
    if (!frame) {
        Serial.println("Camera capture failed");
        return;
    }

    int RESIZE_WIDTH = 28; // Di chuyển khai báo của biến
    int RESIZE_HEIGHT = 28; // Di chuyển khai báo của biến

    // Calculate scale factors for resizing
    int scale_x = frame->width / RESIZE_WIDTH;
    int scale_y = frame->height / RESIZE_HEIGHT;

    // Resize and normalize pixel values
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

            // Calculate average pixel value and normalize
            uint8_t avg_pixel = sum / (scale_x * scale_y);
            float normalized_pixel = avg_pixel / 255.0f;
            input->data.int8[y * RESIZE_WIDTH + x] = (int8_t)(normalized_pixel * 255 - 128);
        }
    }

    esp_camera_fb_return(frame);

    // Run inference
    TfLiteStatus invoke_status = interpreter_ptr->Invoke();
    if (invoke_status != kTfLiteOk) {
        error_reporter->Report("Invoke failed");
        return;
    }

    // Get the output from the inference
    TfLiteTensor* output = interpreter_ptr->output(0);

    int predict = 0;
    int max = -128;  // Adjusted for int8 range

    // Print the output to the Serial
    for (int i = 0; i < output->dims->data[1]; i++) {
        int8_t value = output->data.int8[i];
        if (value > max) {
            max = value;
            predict = i;
        }
    }
    Serial.print("Predict = ");
    Serial.print(predict);
    Serial.println("");
}

void setup(){
    Serial.begin(115200);
    if (!setup_camera(FRAMESIZE_QQVGA)) {
        Serial.println("Camera init failed");
        return;
    }
}

void loop() {
    handle_captureAI();
    delay(2000);
}
