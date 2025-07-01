#include <driver/i2s.h>
#include <HTTPClient.h>

// Hardware pins
#define I2S_WS  15   // Word Select (LRCL)
#define I2S_SD  32   // Serial Data (DOUT)
#define I2S_SCK 14   // Bit Clock (BCLK)

// Audio settings
#define SAMPLE_RATE 44100    // Higher sample rate for better quality
#define BUFFER_SIZE 512      // Smaller buffers for lower latency
#define BIT_DEPTH_24 true    // Enable 24-bit processing

bool recording = false;

void setupCPUFreq() {
  // Set CPU to maximum frequency for stable audio processing (Arduino compatible)
  setCpuFrequencyMhz(240);
}

void setupI2S() {
  const i2s_config_t i2s_config = {
    .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, // 24-bit data in 32-bit container
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S),
    .intr_alloc_flags = 0, // Use default interrupt allocation
    .dma_buf_count = 4,        // Reduced buffer count
    .dma_buf_len = BUFFER_SIZE, // Smaller buffers
    .use_apll = true,          // Use Audio PLL for better clock precision
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  // Install and configure I2S (Arduino compatible error handling)
  if (i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL) != ESP_OK) {
    Serial.println("Failed to install I2S driver");
    return;
  }
  
  if (i2s_set_pin(I2S_NUM_0, &pin_config) != ESP_OK) {
    Serial.println("Failed to set I2S pins");
    return;
  }
  
  // Additional precision settings
  i2s_set_clk(I2S_NUM_0, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO);
  
  Serial.println("I2S configured for high quality audio");
}

// void setupAudioFiltering() {
//   // Configure ADC for better analog performance (Arduino compatible)
//   // Note: These functions are for analog ADC, not I2S mics
//   // analogSetWidth(12);        // 12-bit ADC resolution
//   // analogSetAttenuation(ADC_11db); // Full scale voltage
  
//   // For I2S digital mics, we rely on I2S configuration for quality
//   Serial.println("Audio filtering configured");
// }

void setup() {
  Serial.begin(921600); // Higher baud rate for more data throughput
  
  setupCPUFreq();

  setupI2S();
  
  Serial.println("High-Quality I2S Audio System Ready");
  Serial.println("Commands: 'start' to begin recording, 'stop' to end");
}

void processAudio24Bit(int32_t* buffer, int samples) {
  for (int i = 0; i < samples; i++) {
    // Extract 24-bit audio data (remove upper 8 bits of padding)
    int32_t sample24 = buffer[i] >> 8;
    
    // Option 1: Keep as 24-bit (send 3 bytes)
    if (BIT_DEPTH_24) {
      uint8_t bytes[3];
      bytes[0] = (sample24) & 0xFF;         // LSB
      bytes[1] = (sample24 >> 8) & 0xFF;    // Middle byte
      bytes[2] = (sample24 >> 16) & 0xFF;   // MSB
      Serial.write(bytes, 3);
    }
    // Option 2: High-quality 16-bit conversion
    else {
      // Better 16-bit conversion with dithering
      int16_t sample16 = (sample24 + 128) >> 8; // Add dither + convert
      Serial.write((uint8_t*)&sample16, 2);
    }
  }
}

void loop() {
  // Handle serial commands
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd.equalsIgnoreCase("start")) {
      recording = true;
      Serial.println("START_RECORDING"); // Consistent delimiter
    } 
    else if (cmd.equalsIgnoreCase("stop")) {
      recording = false;
      Serial.println("STOP_RECORDING");
    }
    else if (cmd.equalsIgnoreCase("info")) {
      Serial.printf("Sample Rate: %d Hz, Bit Depth: %s\n", 
                   SAMPLE_RATE, BIT_DEPTH_24 ? "24-bit" : "16-bit");
    }
  }

  // Audio processing loop
  if (recording) {
    int32_t buffer[BUFFER_SIZE];
    size_t bytes_read;
    
    // Read audio with timeout to prevent blocking
    esp_err_t result = i2s_read(I2S_NUM_0, &buffer, sizeof(buffer), 
                               &bytes_read, pdMS_TO_TICKS(100));
    
    if (result == ESP_OK && bytes_read > 0) {
      int samples = bytes_read / sizeof(int32_t);
      processAudio24Bit(buffer, samples);
    }
    
    // Quick command check without blocking
    if (Serial.available()) {
      String cmd = Serial.readStringUntil('\n');
      cmd.trim();
      if (cmd.equalsIgnoreCase("stop")) {
        recording = false;
        Serial.println("STOP_RECORDING");
      }
    }
  }
}