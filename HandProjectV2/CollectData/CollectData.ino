#define CAMERA_MODEL_AI_THINKER
#include <WiFi.h>
#include <WebServer.h>
#include <esp_camera.h>
#include "camera_pin.h"

#define WIFI_SSID "ESP32-CAM-AP"
#define WIFI_PASSWORD "12345678"

const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;

WebServer server(80);

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

void handle_capture() {
    camera_fb_t * frame = esp_camera_fb_get();
    if (!frame) {
        Serial.println("Camera capture failed");
        server.send(500, "text/plain", "Camera capture failed");
        return;
    }

    // Print image width, height, and length
    Serial.print("Orignal Width: ");
    Serial.println(frame->width);
    Serial.print("Orignal Height: ");
    Serial.println(frame->height);

    int RESIZE_WIDTH = 28;
    int RESIZE_HEIGHT = 28;

    // Calculate scale factors for resizing
    int scale_x = frame->width / RESIZE_WIDTH;
    int scale_y = frame->height / RESIZE_HEIGHT;

    String json = "{ \"image\": [";
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
            json += String(avg_pixel);
            Serial.print(avg_pixel);
            if (y != RESIZE_HEIGHT - 1 || x < RESIZE_WIDTH - 1) {
              json += ",";
              Serial.print(",");
            }
        }
        Serial.println();
    }
    json += "] }";
    
    esp_camera_fb_return(frame);

    server.send(200, "application/json", json);
}

void setup() {
    Serial.begin(115200);
    WiFi.softAP(ssid, password);

    IPAddress IP = WiFi.softAPIP();
    Serial.print("AP IP address: ");
    Serial.println(IP);

    if (!setup_camera(FRAMESIZE_QQVGA)) {
        Serial.println("Camera init failed");
        return;
    }

    server.on("/capture", HTTP_GET, handle_capture);
    server.begin();
}

void loop() {
   server.handleClient();
}
