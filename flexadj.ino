#include <WiFi.h>
#include <ThingSpeak.h>

const char* ssid = "pookie";
const char* password = "a4chanaz";

WiFiClient client;

// ThingSpeak
unsigned long channelID = 2989026;
const char* writeAPIKey = "0ZUGUWNDAQXK25KV";

// Pins
const int flexPin = 36;
const int ledPin = 2;

void setup() {
  Serial.begin(115200);
  pinMode(ledPin, OUTPUT);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println(" Connected!");

  ThingSpeak.begin(client);
}

void loop() {
  int flexValue = analogRead(flexPin);
  Serial.println(flexValue);

  // Upload to ThingSpeak
  ThingSpeak.setField(1, flexValue);

  int response = ThingSpeak.writeFields(channelID, writeAPIKey);
  if (response == 200) {
    Serial.println("Data sent to ThingSpeak!");
  } else {
    Serial.print("Failed. Error code: "); Serial.println(response);
  }

  // LED logic
  if (flexValue > 3240) {
    digitalWrite(ledPin, HIGH);
  } else {
    digitalWrite(ledPin, LOW);
  }

  delay(15000); // Wait 15 sec (ThingSpeak rate limit)
}