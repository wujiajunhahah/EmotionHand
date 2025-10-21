using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections;

/// <summary>
/// 校准界面管理器
/// 提供2分钟个性化校准的用户界面
/// </summary>
public class CalibrationUI : MonoBehaviour
{
    [Header("UI组件")]
    [SerializeField] private GameObject calibrationPanel;
    [SerializeField] private TextMeshProUGUI titleText;
    [SerializeField] private TextMeshProUGUI instructionText;
    [SerializeField] private TextMeshProUGUI countdownText;
    [SerializeField] private TextMeshProUGUI progressText;
    [SerializeField] private Button startButton;
    [SerializeField] private Button cancelButton;
    [SerializeField] private Slider progressBar;
    [SerializeField] private Image statusIndicator;

    [Header("校准阶段")]
    [SerializeField] private GameObject[] phaseIndicators; // 阶段指示器

    [Header("颜色配置")]
    [SerializeField] private Color idleColor = Color.gray;
    [SerializeField] private Color activeColor = Color.green;
    [SerializeField] private Color completedColor = Color.blue;
    [SerializeField] private Color errorColor = Color.red;

    // 校准状态
    private enum CalibrationPhase
    {
        Idle,
        RestCalibration,
        GripCalibration,
        GestureCalibration,
        StateCalibration,
        Completed
    }

    private CalibrationPhase currentPhase = CalibrationPhase.Idle;
    private bool isCalibrating = false;
    private float calibrationTimer = 0f;
    private float currentPhaseDuration = 0f;

    // 校准配置
    private float[] phaseDurations = {0f, 60f, 60f, 30f, 30f}; // 各阶段时长
    private string[] phaseNames = {"", "静息状态校准", "轻握状态校准", "手势微调校准", "状态微调校准"};
    private string[] phaseInstructions = {
        "",
        "请保持手臂完全放松，不要有任何肌肉收缩动作。",
        "请保持轻微握拳状态，不要太用力，大约50%的力量。",
        "请按照提示执行指定手势，每个手势保持15秒。",
        "请按照提示进入指定情绪状态，每种状态保持15秒。"
    };

    // 事件
    public event System.Action OnCalibrationStarted;
    public event System.Action OnCalibrationCompleted;
    public event System.Action<string> OnCalibrationError;

    void Start()
    {
        InitializeUI();
        ResetCalibration();
    }

    /// <summary>
    /// 初始化UI
    /// </summary>
    private void InitializeUI()
    {
        // 设置按钮事件
        if (startButton != null)
        {
            startButton.onClick.AddListener(StartCalibration);
        }

        if (cancelButton != null)
        {
            cancelButton.onClick.AddListener(CancelCalibration);
        }

        // 初始化进度条
        if (progressBar != null)
        {
            progressBar.minValue = 0f;
            progressBar.maxValue = 100f;
            progressBar.value = 0f;
        }

        // 隐藏校准面板
        if (calibrationPanel != null)
        {
            calibrationPanel.SetActive(false);
        }

        // 初始化阶段指示器
        UpdatePhaseIndicators();
    }

    /// <summary>
    /// 开始校准
    /// </summary>
    public void StartCalibration()
    {
        if (isCalibrating)
            return;

        Debug.Log("开始校准流程");
        OnCalibrationStarted?.Invoke();

        isCalibrating = true;
        currentPhase = CalibrationPhase.RestCalibration;
        calibrationTimer = 0f;
        currentPhaseDuration = phaseDurations[(int)currentPhase];

        // 更新UI
        if (calibrationPanel != null)
        {
            calibrationPanel.SetActive(true);
        }

        if (startButton != null)
        {
            startButton.interactable = false;
        }

        if (cancelButton != null)
        {
            cancelButton.interactable = true;
        }

        UpdateUI();
        UpdatePhaseIndicators();

        // 启动校准协程
        StartCoroutine(CalibrationCoroutine());
    }

    /// <summary>
    /// 取消校准
    /// </summary>
    public void CancelCalibration()
    {
        if (!isCalibrating)
            return;

        Debug.Log("取消校准");
        isCalibrating = false;
        StopAllCoroutines();
        ResetCalibration();
    }

    /// <summary>
    /// 校准协程
    /// </summary>
    private IEnumerator CalibrationCoroutine()
    {
        while (isCalibrating && currentPhase != CalibrationPhase.Completed)
        {
            calibrationTimer += Time.deltaTime;

            // 更新进度
            float phaseProgress = calibrationTimer / currentPhaseDuration;
            UpdateProgress(phaseProgress);

            // 检查当前阶段是否完成
            if (calibrationTimer >= currentPhaseDuration)
            {
                CompleteCurrentPhase();
            }

            // 更新倒计时
            UpdateCountdown();

            yield return null;
        }

        if (currentPhase == CalibrationPhase.Completed)
        {
            CompleteCalibration();
        }
    }

    /// <summary>
    /// 完成当前阶段
    /// </summary>
    private void CompleteCurrentPhase()
    {
        Debug.Log($"完成校准阶段: {currentPhase}");

        // 进入下一阶段
        currentPhase++;

        if (currentPhase <= CalibrationPhase.StateCalibration)
        {
            // 重置计时器
            calibrationTimer = 0f;
            currentPhaseDuration = phaseDurations[(int)currentPhase];

            // 更新UI
            UpdateUI();
            UpdatePhaseIndicators();

            // 特殊处理手势和状态校准
            if (currentPhase == CalibrationPhase.GestureCalibration)
            {
                StartCoroutine(HandleGestureCalibration());
            }
            else if (currentPhase == CalibrationPhase.StateCalibration)
            {
                StartCoroutine(HandleStateCalibration());
            }
        }
        else
        {
            currentPhase = CalibrationPhase.Completed;
        }
    }

    /// <summary>
    /// 处理手势校准
    /// </summary>
    private IEnumerator HandleGestureCalibration()
    {
        string[] gestures = {"Fist", "Open"};
        float gestureDuration = 15f; // 每个手势15秒

        for (int i = 0; i < gestures.Length; i++)
        {
            string gesture = gestures[i];
            instructionText.text = $"请执行 '{gesture}' 手势并保持({gestureDuration}秒)";

            float gestureTimer = 0f;
            while (gestureTimer < gestureDuration && isCalibrating)
            {
                gestureTimer += Time.deltaTime;
                float progress = gestureTimer / gestureDuration;
                UpdateProgress(progress);
                UpdateCountdown(gestureDuration - gestureTimer);
                yield return null;
            }

            if (!isCalibrating)
                break;
        }

        // 手势校准完成
        CompleteCurrentPhase();
    }

    /// <summary>
    /// 处理状态校准
    /// </summary>
    private IEnumerator HandleStateCalibration()
    {
        string[] states = {"Relaxed", "Focused"};
        float stateDuration = 15f; // 每种状态15秒

        for (int i = 0; i < states.Length; i++)
        {
            string state = states[i];
            instructionText.text = $"请进入 '{state}' 状态并保持({stateDuration}秒)";

            float stateTimer = 0f;
            while (stateTimer < stateDuration && isCalibrating)
            {
                stateTimer += Time.deltaTime;
                float progress = stateTimer / stateDuration;
                UpdateProgress(progress);
                UpdateCountdown(stateDuration - stateTimer);
                yield return null;
            }

            if (!isCalibrating)
                break;
        }

        // 状态校准完成
        CompleteCurrentPhase();
    }

    /// <summary>
    /// 完成校准
    /// </summary>
    private void CompleteCalibration()
    {
        isCalibrating = false;
        Debug.Log("校准完成！");

        OnCalibrationCompleted?.Invoke();

        // 显示完成状态
        titleText.text = "校准完成！";
        instructionText.text = "个性化校准已完成，系统现在可以更好地识别您的手势和情绪状态。";

        if (statusIndicator != null)
        {
            statusIndicator.color = completedColor;
        }

        if (progressBar != null)
        {
            progressBar.value = 100f;
        }

        if (countdownText != null)
        {
            countdownText.text = "完成";
        }

        // 3秒后自动关闭面板
        StartCoroutine(AutoClosePanel());
    }

    /// <summary>
    /// 自动关闭面板
    /// </summary>
    private IEnumerator AutoClosePanel()
    {
        yield return new WaitForSeconds(3f);

        if (calibrationPanel != null)
        {
            calibrationPanel.SetActive(false);
        }

        ResetCalibration();
    }

    /// <summary>
    /// 更新UI
    /// </summary>
    private void UpdateUI()
    {
        if (titleText != null && currentPhase > 0)
        {
            titleText.text = phaseNames[(int)currentPhase];
        }

        if (instructionText != null && currentPhase > 0)
        {
            instructionText.text = phaseInstructions[(int)currentPhase];
        }

        // 更新状态指示器颜色
        if (statusIndicator != null)
        {
            statusIndicator.color = activeColor;
        }
    }

    /// <summary>
    /// 更新进度
    /// </summary>
    private void UpdateProgress(float phaseProgress)
    {
        if (progressBar != null)
        {
            // 计算总体进度
            float totalProgress = (currentPhase - 1 + phaseProgress) / (phaseDurations.Length - 1);
            progressBar.value = totalProgress * 100f;
        }

        if (progressText != null)
        {
            float totalProgress = (currentPhase - 1 + phaseProgress) / (phaseDurations.Length - 1);
            progressText.text = $"总体进度: {totalProgress * 100:F1}%";
        }
    }

    /// <summary>
    /// 更新倒计时
    /// </summary>
    private void UpdateCountdown(float remainingTime = -1f)
    {
        if (countdownText != null)
        {
            if (remainingTime < 0)
            {
                remainingTime = currentPhaseDuration - calibrationTimer;
            }

            countdownText.text = $"剩余时间: {remainingTime:F1}秒";
        }
    }

    /// <summary>
    /// 更新阶段指示器
    /// </summary>
    private void UpdatePhaseIndicators()
    {
        if (phaseIndicators == null)
            return;

        for (int i = 0; i < phaseIndicators.Length; i++)
        {
            if (phaseIndicators[i] != null)
            {
                Image indicator = phaseIndicators[i].GetComponent<Image>();
                if (indicator != null)
                {
                    if (i < (int)currentPhase)
                    {
                        indicator.color = completedColor;
                    }
                    else if (i == (int)currentPhase && isCalibrating)
                    {
                        indicator.color = activeColor;
                    }
                    else
                    {
                        indicator.color = idleColor;
                    }
                }
            }
        }
    }

    /// <summary>
    /// 重置校准
    /// </summary>
    private void ResetCalibration()
    {
        currentPhase = CalibrationPhase.Idle;
        isCalibrating = false;
        calibrationTimer = 0f;
        currentPhaseDuration = 0f;

        // 重置UI
        if (titleText != null)
        {
            titleText.text = "个性化校准";
        }

        if (instructionText != null)
        {
            instructionText.text = "校准将帮助系统更好地识别您的个人特征。\n\n总共需要约2分钟，请确保传感器正确佩戴。";
        }

        if (countdownText != null)
        {
            countdownText.text = "";
        }

        if (progressText != null)
        {
            progressText.text = "准备就绪";
        }

        if (progressBar != null)
        {
            progressBar.value = 0f;
        }

        if (statusIndicator != null)
        {
            statusIndicator.color = idleColor;
        }

        if (startButton != null)
        {
            startButton.interactable = true;
        }

        if (cancelButton != null)
        {
            cancelButton.interactable = false;
        }

        UpdatePhaseIndicators();
    }

    /// <summary>
    /// 显示校准错误
    /// </summary>
    public void ShowError(string errorMessage)
    {
        Debug.LogError($"校准错误: {errorMessage}");
        OnCalibrationError?.Invoke(errorMessage);

        if (titleText != null)
        {
            titleText.text = "校准错误";
        }

        if (instructionText != null)
        {
            instructionText.text = errorMessage;
        }

        if (statusIndicator != null)
        {
            statusIndicator.color = errorColor;
        }

        isCalibrating = false;
        StopAllCoroutines();

        // 5秒后重置
        StartCoroutine(ResetAfterError());
    }

    /// <summary>
    /// 错误后重置
    /// </summary>
    private IEnumerator ResetAfterError()
    {
        yield return new WaitForSeconds(5f);
        ResetCalibration();
    }

    /// <summary>
    /// 显示/隐藏校准面板
    /// </summary>
    public void ShowCalibrationPanel(bool show)
    {
        if (calibrationPanel != null)
        {
            calibrationPanel.SetActive(show);
        }
    }

    /// <summary>
    /// 获取校准状态
    /// </summary>
    public bool IsCalibrating()
    {
        return isCalibrating;
    }

    /// <summary>
    /// 获取当前阶段
    /// </summary>
    public string GetCurrentPhase()
    {
        return currentPhase.ToString();
    }

    void OnDestroy()
    {
        // 清理事件
        if (startButton != null)
        {
            startButton.onClick.RemoveAllListeners();
        }

        if (cancelButton != null)
        {
            cancelButton.onClick.RemoveAllListeners();
        }
    }
}