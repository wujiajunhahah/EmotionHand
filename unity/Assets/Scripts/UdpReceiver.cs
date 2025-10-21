using UnityEngine;
using System;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

/// <summary>
/// UDP数据接收器
/// 接收来自Python推理管线的实时数据
/// </summary>
public class UdpReceiver : MonoBehaviour
{
    [Header("网络配置")]
    [SerializeField] private string ipAddress = "127.0.0.1";
    [SerializeField] private int port = 9001;

    [Header("调试信息")]
    [SerializeField] private bool showDebugInfo = true;
    [SerializeField] private int maxDataHistory = 100;

    // 网络相关
    private UdpClient udpClient;
    private Thread receiveThread;
    private bool isReceiving = false;

    // 数据结构
    [System.Serializable]
    public struct EmotionData
    {
        public string gesture;
        public string state;
        public float confidence;
        public float latency;
        public float[] features;
        public long timestamp;
    }

    // 数据缓存
    private EmotionData currentData;
    private EmotionData[] dataHistory;
    private int historyIndex = 0;

    // 事件
    public event Action<EmotionData> OnDataReceived;
    public event Action<string, float> OnGestureChanged;
    public event Action<string, float> OnStateChanged;

    // 属性
    public EmotionData CurrentData => currentData;
    public EmotionData[] DataHistory => dataHistory;
    public bool IsReceiving => isReceiving;
    public int DataCount { get; private set; }

    void Start()
    {
        InitializeReceiver();
    }

    void OnDestroy()
    {
        StopReceiver();
    }

    void Update()
    {
        // 在主线程中触发事件
        if (DataCount > 0)
        {
            OnDataReceived?.Invoke(currentData);
        }
    }

    /// <summary>
    /// 初始化UDP接收器
    /// </summary>
    private void InitializeReceiver()
    {
        try
        {
            // 初始化数据历史
            dataHistory = new EmotionData[maxDataHistory];

            // 创建UDP客户端
            udpClient = new UdpClient(port);
            udpClient.Client.ReceiveBufferSize = 1024;

            // 启动接收线程
            isReceiving = true;
            receiveThread = new Thread(ReceiveData);
            receiveThread.IsBackground = true;
            receiveThread.Start();

            if (showDebugInfo)
            {
                Debug.Log($"UDP接收器启动成功 - {ipAddress}:{port}");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"UDP接收器初始化失败: {e.Message}");
        }
    }

    /// <summary>
    /// 停止接收器
    /// </summary>
    private void StopReceiver()
    {
        isReceiving = false;

        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join(1000);
        }

        if (udpClient != null)
        {
            udpClient.Close();
            udpClient = null;
        }

        if (showDebugInfo)
        {
            Debug.Log("UDP接收器已停止");
        }
    }

    /// <summary>
    /// 接收数据线程
    /// </summary>
    private void ReceiveData()
    {
        IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);

        while (isReceiving)
        {
            try
            {
                if (udpClient != null && udpClient.Available > 0)
                {
                    byte[] data = udpClient.Receive(ref remoteEndPoint);
                    string message = Encoding.UTF8.GetString(data);

                    // 解析数据
                    EmotionData emotionData = ParseData(message);

                    // 更新当前数据
                    currentData = emotionData;

                    // 添加到历史记录
                    dataHistory[historyIndex] = emotionData;
                    historyIndex = (historyIndex + 1) % maxDataHistory;
                    DataCount++;

                    // 触发事件
                    OnGestureChanged?.Invoke(emotionData.gesture, emotionData.confidence);
                    OnStateChanged?.Invoke(emotionData.state, emotionData.confidence);

                    if (showDebugInfo && DataCount % 50 == 0)
                    {
                        Debug.Log($"收到数据 #{DataCount}: {emotionData.gesture} | {emotionData.state} | 置信度: {emotionData.confidence:F3} | 延迟: {emotionData.latency:F1}ms");
                    }
                }
                else
                {
                    Thread.Sleep(1);
                }
            }
            catch (Exception e)
            {
                if (isReceiving) // 只在仍在接收时记录错误
                {
                    Debug.LogError($"UDP数据接收错误: {e.Message}");
                }
            }
        }
    }

    /// <summary>
    /// 解析接收到的数据
    /// 格式: "手势|状态|置信度|延迟"
    /// </summary>
    private EmotionData ParseData(string message)
    {
        EmotionData data = new EmotionData();

        try
        {
            string[] parts = message.Split('|');

            if (parts.Length >= 4)
            {
                data.gesture = parts[0];
                data.state = parts[1];
                data.confidence = float.Parse(parts[1]);
                data.latency = float.Parse(parts[2]);
                data.timestamp = DateTimeOffset.Now.ToUnixTimeMilliseconds();
            }
            else
            {
                // 解析失败，使用默认值
                data.gesture = "Unknown";
                data.state = "Unknown";
                data.confidence = 0.0f;
                data.latency = 0.0f;
                data.timestamp = DateTimeOffset.Now.ToUnixTimeMilliseconds();
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"数据解析错误: {e.Message}");
            data.gesture = "Error";
            data.state = "Error";
            data.confidence = 0.0f;
            data.latency = 0.0f;
            data.timestamp = DateTimeOffset.Now.ToUnixTimeMilliseconds();
        }

        return data;
    }

    /// <summary>
    /// 获取最近的数据平均值
    /// </summary>
    public EmotionData GetAverageData(int count = 10)
    {
        if (DataCount == 0)
            return currentData;

        int actualCount = Mathf.Min(count, DataCount, maxDataHistory);
        EmotionData[] recentData = new EmotionData[actualCount];

        for (int i = 0; i < actualCount; i++)
        {
            int index = (historyIndex - 1 - i + maxDataHistory) % maxDataHistory;
            recentData[i] = dataHistory[index];
        }

        // 计算平均值
        EmotionData average = new EmotionData();
        average.gesture = GetMostCommonGesture(recentData);
        average.state = GetMostCommonState(recentData);
        average.confidence = GetAverageConfidence(recentData);
        average.latency = GetAverageLatency(recentData);
        average.timestamp = DateTimeOffset.Now.ToUnixTimeMilliseconds();

        return average;
    }

    private string GetMostCommonGesture(EmotionData[] data)
    {
        var gestureCounts = new System.Collections.Generic.Dictionary<string, int>();

        foreach (var d in data)
        {
            if (gestureCounts.ContainsKey(d.gesture))
                gestureCounts[d.gesture]++;
            else
                gestureCounts[d.gesture] = 1;
        }

        string mostCommon = "";
        int maxCount = 0;

        foreach (var kvp in gestureCounts)
        {
            if (kvp.Value > maxCount)
            {
                maxCount = kvp.Value;
                mostCommon = kvp.Key;
            }
        }

        return mostCommon;
    }

    private string GetMostCommonState(EmotionData[] data)
    {
        var stateCounts = new System.Collections.Generic.Dictionary<string, int>();

        foreach (var d in data)
        {
            if (stateCounts.ContainsKey(d.state))
                stateCounts[d.state]++;
            else
                stateCounts[d.state] = 1;
        }

        string mostCommon = "";
        int maxCount = 0;

        foreach (var kvp in stateCounts)
        {
            if (kvp.Value > maxCount)
            {
                maxCount = kvp.Value;
                mostCommon = kvp.Key;
            }
        }

        return mostCommon;
    }

    private float GetAverageConfidence(EmotionData[] data)
    {
        float sum = 0f;
        foreach (var d in data)
            sum += d.confidence;
        return sum / data.Length;
    }

    private float GetAverageLatency(EmotionData[] data)
    {
        float sum = 0f;
        foreach (var d in data)
            sum += d.latency;
        return sum / data.Length;
    }

    /// <summary>
    /// 获取连接状态信息
    /// </summary>
    public string GetStatusInfo()
    {
        if (!isReceiving)
            return "接收器已停止";

        if (DataCount == 0)
            return "等待数据...";

        var avgData = GetAverageData(20);
        return $"数据: {DataCount} | 手势: {avgData.gesture} | 状态: {avgData.state} | 置信度: {avgData.confidence:F3} | 延迟: {avgData.latency:F1}ms";
    }

    /// <summary>
    /// 重置数据历史
    /// </summary>
    public void ResetHistory()
    {
        dataHistory = new EmotionData[maxDataHistory];
        historyIndex = 0;
        DataCount = 0;
        currentData = new EmotionData();
    }
}