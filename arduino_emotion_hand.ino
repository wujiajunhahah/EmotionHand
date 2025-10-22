/*
 * EmotionHand Arduino代码
 * 读取EMG和GSR传感器数据并通过串口发送
 *
 * 硬件连接：
 * - EMG传感器 (Muscle Sensor v3) -> A0
 * - GSR传感器 (指套式) -> A1
 * - 可选：LED指示灯 -> D13
 *
 * 数据格式：emg_value,gsr_value
 * 更新频率：50Hz (每20ms)
 */

// 传感器引脚定义
const int EMG_PIN = A0;    // EMG传感器连接到A0
const int GSR_PIN = A1;    // GSR传感器连接到A1
const int LED_PIN = 13;    // 状态指示LED

// 数据处理参数
const int SAMPLE_RATE = 50;        // 采样率 (Hz)
const int SAMPLE_INTERVAL = 20;    // 采样间隔 (ms)
const int EMG_FILTER_ALPHA = 0.8;  // EMG低通滤波系数
const int GSR_FILTER_ALPHA = 0.9;  // GSR低通滤波系数

// 数据变量
float emg_raw = 0;
float gsr_raw = 0;
float emg_filtered = 0;
float gsr_filtered = 0;
unsigned long last_time = 0;

// 校准参数
bool calibrated = false;
float emg_baseline = 0;
float gsr_baseline = 0;
int calibration_count = 0;
const int CALIBRATION_SAMPLES = 1000;  // 20秒校准

void setup() {
  // 初始化串口
  Serial.begin(115200);
  while (!Serial) {
    ; // 等待串口连接
  }

  // 设置引脚模式
  pinMode(EMG_PIN, INPUT);
  pinMode(GSR_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);

  // LED闪烁表示启动
  blinkLED(3, 200);

  Serial.println("EmotionHand Arduino System");
  Serial.println("Initializing sensors...");

  delay(1000);
  last_time = millis();
}

void loop() {
  unsigned long current_time = millis();

  // 固定频率采样
  if (current_time - last_time >= SAMPLE_INTERVAL) {
    last_time = current_time;

    // 读取传感器数据
    readSensors();

    // 校准或处理数据
    if (!calibrated) {
      performCalibration();
    } else {
      processData();
    }
  }

  // 处理串口命令
  handleCommands();
}

void readSensors() {
  // 读取原始数据
  emg_raw = analogRead(EMG_PIN);
  gsr_raw = analogRead(GSR_PIN);
}

void performCalibration() {
  // 收集校准数据
  static unsigned long calibration_start = millis();

  emg_baseline += emg_raw;
  gsr_baseline += gsr_raw;
  calibration_count++;

  // LED快闪表示校准中
  digitalWrite(LED_PIN, (millis() / 100) % 2);

  // 检查校准完成
  if (calibration_count >= CALIBRATION_SAMPLES) {
    emg_baseline /= CALIBRATION_SAMPLES;
    gsr_baseline /= CALIBRATION_SAMPLES;
    calibrated = true;

    Serial.println("Calibration completed!");
    Serial.print("EMG baseline: ");
    Serial.println(emg_baseline);
    Serial.print("GSR baseline: ");
    Serial.println(gsr_baseline);

    // LED常亮表示校准完成
    digitalWrite(LED_PIN, HIGH);
    delay(1000);
    digitalWrite(LED_PIN, LOW);
  } else {
    // 显示校准进度
    if (calibration_count % 100 == 0) {
      int progress = (calibration_count * 100) / CALIBRATION_SAMPLES;
      Serial.print("Calibrating... ");
      Serial.print(progress);
      Serial.println("%");
    }
  }
}

void processData() {
  // 应用低通滤波器
  emg_filtered = EMG_FILTER_ALPHA * emg_filtered + (1 - EMG_FILTER_ALPHA) * emg_raw;
  gsr_filtered = GSR_FILTER_ALPHA * gsr_filtered + (1 - GSR_FILTER_ALPHA) * gsr_raw;

  // 去除基线偏移
  float emg_corrected = emg_filtered - emg_baseline;
  float gsr_corrected = gsr_filtered - gsr_baseline;

  // 限制范围
  emg_corrected = constrain(emg_corrected, -500, 500);
  gsr_corrected = constrain(gsr_corrected, -200, 200);

  // 映射到正范围
  emg_corrected = map(emg_corrected, -500, 500, 0, 1023);
  gsr_corrected = map(gsr_corrected, -200, 200, 0, 1023);

  // 确保在有效范围内
  emg_corrected = constrain(emg_corrected, 0, 1023);
  gsr_corrected = constrain(gsr_corrected, 0, 1023);

  // 发送数据 (格式：emg,gsr)
  Serial.print(emg_corrected);
  Serial.print(",");
  Serial.println(gsr_corrected);

  // 根据EMG活动控制LED
  if (emg_corrected > 600) {
    digitalWrite(LED_PIN, HIGH);
  } else {
    digitalWrite(LED_PIN, LOW);
  }
}

void handleCommands() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "RESET") {
      // 重置校准
      calibrated = false;
      calibration_count = 0;
      emg_baseline = 0;
      gsr_baseline = 0;
      emg_filtered = 0;
      gsr_filtered = 0;
      Serial.println("Calibration reset");
    }
    else if (command == "STATUS") {
      // 发送状态信息
      Serial.print("Status: ");
      Serial.println(calibrated ? "Calibrated" : "Calibrating");
      Serial.print("EMG: ");
      Serial.print(emg_raw);
      Serial.print(" -> ");
      Serial.println(emg_filtered);
      Serial.print("GSR: ");
      Serial.print(gsr_raw);
      Serial.print(" -> ");
      Serial.println(gsr_filtered);
    }
    else if (command == "INFO") {
      // 发送硬件信息
      Serial.println("EmotionHand Arduino Controller");
      Serial.print("Sample Rate: ");
      Serial.print(SAMPLE_RATE);
      Serial.println(" Hz");
      Serial.print("EMG Pin: A");
      Serial.println(EMG_PIN - A0);
      Serial.print("GSR Pin: A");
      Serial.println(GSR_PIN - A0);
    }
  }
}

void blinkLED(int times, int duration) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(duration);
    digitalWrite(LED_PIN, LOW);
    delay(duration);
  }
}