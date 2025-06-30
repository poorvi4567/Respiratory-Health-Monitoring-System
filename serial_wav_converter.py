# import serial
# import wave
# import time
# import datetime
# import numpy as np
# import scipy.signal as signal
# import os

# # === CONFIGURATION ===
# SERIAL_PORT = 'COM7'           # Change this to match your port
# BAUD_RATE = 115200
# DURATION = 10                  # seconds
# SAMPLE_RATE = 7000             # effective sample rate based on your system

# # === FILE NAME ===
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# OUTPUT_FILE = f"C:/Users/poorv/Downloads/recorded_breath_{timestamp}.wav"

# # === SERIAL SETUP ===
# ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
# time.sleep(2)  # Allow ESP32 to reboot
# print("📡 Serial connected.")

# # === START RECORDING ===
# ser.write(b'start\n')
# print("🎙️  STARTING recording...")

# start_time = time.time()
# audio_data = bytearray()

# # === READ SERIAL AUDIO STREAM FOR 20 SECONDS ===
# while time.time() - start_time < DURATION:
#     if ser.in_waiting:
#         audio_data += ser.read(ser.in_waiting)

# ser.write(b'stop\n')
# print("🛑 STOPPED recording.")
# ser.close()

# # === CHECK SIZE ===
# print(f"📦 Total bytes received: {len(audio_data)}")
# if len(audio_data) % 2 != 0:
#     audio_data = audio_data[:-1]  # trim last byte if odd

# # === CONVERT TO int16 AND FILTER ===
# samples = np.frombuffer(audio_data, dtype=np.int16)

# # High-pass filter at 150Hz
# b, a = signal.butter(4, 150 / (SAMPLE_RATE / 2), btype='highpass')
# filtered = signal.filtfilt(b, a, samples)

# # Normalize
# filtered = filtered / np.max(np.abs(filtered))
# filtered = (filtered * 32767).astype(np.int16)


# # === FILTER AND NORMALIZE ===
# samples = np.frombuffer(audio_data, dtype=np.int16)

# # High-pass filter: cutoff at 150 Hz
# b, a = signal.butter(4, 150 / (SAMPLE_RATE / 2), btype='highpass')
# filtered = signal.filtfilt(b, a, samples)

# # Normalize to int16 range
# filtered = filtered / np.max(np.abs(filtered))  # range -1 to 1
# filtered = (filtered * 32767).astype(np.int16)

# # === SAVE TO .WAV ===
# # Just save raw audio_data directly (no filter, no normalize)
# with wave.open(OUTPUT_FILE, 'wb') as wf:
#     wf.setnchannels(1)
#     wf.setsampwidth(2)
#     wf.setframerate(SAMPLE_RATE)  # ← matches actual sample rate
#     wf.writeframes(audio_data)


# print(f"✅ Saved cleaned WAV: {OUTPUT_FILE}")
#!/usr/bin/env python3
#!/usr/bin/env python3
"""
ESP32 I2S Audio Recorder
Captures audio from ESP32 I2S microphone for specified duration
Supports both 16-bit and 24-bit audio formats
"""

import serial
import time
import numpy as np
import wave
import struct
import argparse
from datetime import datetime
import os

class ESP32AudioRecorder:
    def __init__(self, port='COM7', baud_rate=921600, timeout=1):
        """
        Initialize ESP32 Audio Recorder
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baud_rate: Serial communication speed (must match ESP32 code)
            timeout: Serial read timeout in seconds
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_conn = None
        self.is_recording = False
        
        # Audio configuration (must match ESP32 settings)
        self.sample_rate = 44100  # Hz
        self.bit_depth_24 = True  # Set to False for 16-bit mode
        self.channels = 1  # Mono
        
    def connect(self):
        """Establish serial connection to ESP32"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            time.sleep(2)  # Allow ESP32 to reset
            print(f"Connected to ESP32 on {self.port} at {self.baud_rate} baud")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial_conn and self.serial_conn.is_open:
            if self.is_recording:
                self.stop_recording()
            self.serial_conn.close()
            print("Disconnected from ESP32")
    
    def send_command(self, command):
        """Send command to ESP32"""
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Error: Not connected to ESP32")
            return False
        
        try:
            self.serial_conn.write(f"{command}\n".encode())
            time.sleep(0.1)  # Allow ESP32 to process command
            
            # Read response
            response = self.serial_conn.readline().decode().strip()
            print(f"ESP32 response: {response}")
            return True
        except Exception as e:
            print(f"Error sending command: {e}")
            return False
    
    def start_recording(self):
        """Start audio recording on ESP32"""
        if self.send_command("start"):
            self.is_recording = True
            print("Recording started...")
            return True
        return False
    
    def stop_recording(self):
        """Stop audio recording on ESP32"""
        if self.send_command("stop"):
            self.is_recording = False
            print("Recording stopped")
            return True
        return False
    
    def record_audio(self, duration_seconds=10, output_filename=None):
        """
        Record audio for specified duration
        
        Args:
            duration_seconds: Recording duration in seconds
            output_filename: Output WAV file name (auto-generated if None)
        
        Returns:
            tuple: (audio_data, filename) or (None, None) if failed
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Error: Not connected to ESP32")
            return None, None
        
        # Generate filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bit_depth_str = "24bit" if self.bit_depth_24 else "16bit"
            output_filename = f"esp32_audio_{timestamp}_{bit_depth_str}.wav"
        
        print(f"Recording {duration_seconds} seconds of audio...")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Bit depth: {'24-bit' if self.bit_depth_24 else '16-bit'}")
        print(f"Output file: {output_filename}")
        
        # Clear any existing data in serial buffer
        self.serial_conn.reset_input_buffer()
        
        # Start recording
        if not self.start_recording():
            return None, None
        
        # Calculate expected data
        bytes_per_sample = 3 if self.bit_depth_24 else 2
        total_samples = int(self.sample_rate * duration_seconds)
        expected_bytes = total_samples * bytes_per_sample
        
        print(f"Expected samples: {total_samples}")
        print(f"Expected bytes: {expected_bytes}")
        
        # Record audio data
        audio_data = bytearray()
        start_time = time.time()
        last_update = start_time
        
        try:
            while len(audio_data) < expected_bytes:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check if duration exceeded
                if elapsed >= duration_seconds:
                    break
                
                # Progress update every second
                if current_time - last_update >= 1.0:
                    progress = (len(audio_data) / expected_bytes) * 100
                    print(f"Progress: {progress:.1f}% - {elapsed:.1f}s/{duration_seconds}s")
                    last_update = current_time
                
                # Read available data
                if self.serial_conn.in_waiting > 0:
                    chunk = self.serial_conn.read(self.serial_conn.in_waiting)
                    audio_data.extend(chunk)
                else:
                    time.sleep(0.001)  # Small delay to prevent busy waiting
            
        except KeyboardInterrupt:
            print("\nRecording interrupted by user")
        except Exception as e:
            print(f"Error during recording: {e}")
            return None, None
        finally:
            # Always stop recording
            self.stop_recording()
        
        # Process and save audio
        if len(audio_data) > 0:
            print(f"Recorded {len(audio_data)} bytes of audio data")
            
            # Convert to numpy array and save as WAV
            audio_array = self.process_audio_data(audio_data)
            if audio_array is not None:
                self.save_wav_file(audio_array, output_filename)
                print(f"Audio saved to: {output_filename}")
                return audio_array, output_filename
        
        print("No audio data recorded")
        return None, None
    
    def process_audio_data(self, raw_data):
        """Convert raw bytes to numpy audio array"""
        try:
            if self.bit_depth_24:
                # Process 24-bit audio (3 bytes per sample)
                if len(raw_data) % 3 != 0:
                    # Trim to multiple of 3
                    raw_data = raw_data[:-(len(raw_data) % 3)]
                
                samples = []
                for i in range(0, len(raw_data), 3):
                    # Reconstruct 24-bit signed integer
                    sample = struct.unpack('<I', raw_data[i:i+3] + b'\x00')[0]
                    if sample & 0x800000:  # Sign extension for negative numbers
                        sample |= 0xFF000000
                    samples.append(struct.unpack('<i', struct.pack('<I', sample))[0])
                
                # Convert to numpy array and normalize to float32
                audio_array = np.array(samples, dtype=np.int32)
                audio_array = audio_array.astype(np.float32) / (2**23)
                
            else:
                # Process 16-bit audio (2 bytes per sample)
                if len(raw_data) % 2 != 0:
                    raw_data = raw_data[:-1]  # Remove last byte if odd
                
                # Convert to 16-bit signed integers
                samples = struct.unpack(f'<{len(raw_data)//2}h', raw_data)
                audio_array = np.array(samples, dtype=np.int16)
                audio_array = audio_array.astype(np.float32) / (2**15)
            
            print(f"Processed {len(audio_array)} audio samples")
            return audio_array
            
        except Exception as e:
            print(f"Error processing audio data: {e}")
            return None
    
    def save_wav_file(self, audio_array, filename):
        """Save audio array as WAV file"""
        try:
            # Convert back to appropriate bit depth for WAV file
            if self.bit_depth_24:
                # Save as 24-bit WAV
                audio_int = (audio_array * (2**23 - 1)).astype(np.int32)
                sample_width = 3
            else:
                # Save as 16-bit WAV
                audio_int = (audio_array * (2**15 - 1)).astype(np.int16)
                sample_width = 2
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(self.sample_rate)
                
                if self.bit_depth_24:
                    # Pack 24-bit samples
                    packed_data = b''
                    for sample in audio_int:
                        packed_data += struct.pack('<i', sample)[:3]  # Take only 3 bytes
                    wav_file.writeframes(packed_data)
                else:
                    wav_file.writeframes(audio_int.tobytes())
            
            print(f"WAV file saved successfully: {filename}")
            
        except Exception as e:
            print(f"Error saving WAV file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Record audio from ESP32 I2S microphone')
    parser.add_argument('--port', '-p', default='COM7', 
                       help='Serial port (default: COM7)')
    parser.add_argument('--duration', '-d', type=int, default=10,
                       help='Recording duration in seconds (default: 10)')
    parser.add_argument('--output', '-o', 
                       help='Output filename (auto-generated if not specified)')
    parser.add_argument('--baud', '-b', type=int, default=921600,
                       help='Serial baud rate (default: 921600)')
    parser.add_argument('--16bit', action='store_true', dest='bit16',
                       help='Use 16-bit mode instead of 24-bit')
    
    args = parser.parse_args()
    
    # Create recorder instance
    recorder = ESP32AudioRecorder(
        port=args.port,
        baud_rate=args.baud
    )
    
    # Set bit depth based on argument
    recorder.bit_depth_24 = not args.bit16
    
    try:
        # Connect to ESP32
        if not recorder.connect():
            return
        
        # Record audio
        audio_data, filename = recorder.record_audio(
            duration_seconds=args.duration,
            output_filename=args.output
        )
        
        if audio_data is not None:
            print(f"\nRecording completed successfully!")
            print(f"File: {filename}")
            print(f"Duration: {len(audio_data) / recorder.sample_rate:.2f} seconds")
            print(f"Sample rate: {recorder.sample_rate} Hz")
            print(f"Bit depth: {'24-bit' if recorder.bit_depth_24 else '16-bit'}")
        else:
            print("Recording failed!")
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        recorder.disconnect()

if __name__ == "__main__":
    main()