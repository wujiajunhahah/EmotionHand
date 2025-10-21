using UnityEngine;
using System.Collections;
using System.Collections.Generic;

/// <summary>
/// EmotionHand 3D可视化器
/// 根据EMG+GSR数据实时渲染手部状态和情绪效果
/// </summary>
public class EmotionHandVisualizer : MonoBehaviour
{
    [Header("手部模型")]
    [SerializeField] private SkinnedMeshRenderer handRenderer;
    [SerializeField] private Transform[] fingerBones;
    [SerializeField] private Transform wristTransform;

    [Header("材质配置")]
    [SerializeField] private Material defaultMaterial;
    [SerializeField] private Material relaxedMaterial;
    [SerializeField] private Material focusedMaterial;
    [SerializeField] private Material stressedMaterial;
    [SerializeField] private Material fatiguedMaterial;

    [Header("视觉效果")]
    [SerializeField] private ParticleSystem emotionParticles;
    [SerializeField] private Light handLight;
    [SerializeField] private TrailRenderer[] fingerTrails;

    [Header("状态颜色")]
    [SerializeField] private Color relaxedColor = Color.blue;
    [SerializeField] private Color focusedColor = Color.green;
    [SerializeField] private Color stressedColor = Color.red;
    [SerializeField] private Color fatiguedColor = Color.yellow;
    [SerializeField] private Color neutralColor = Color.white;

    [Header("动画参数")]
    [SerializeField] private float transitionSpeed = 2.0f;
    [SerializeField] private float particleEmissionRate = 50f;
    [SerializeField] private float lightIntensityMultiplier = 1.5f;

    [Header("手势配置")]
    [SerializeField] private float[] fistBendAngles = {45f, 60f, 70f, 80f, 90f}; // 各手指弯曲角度
    [SerializeField] private float[] openBendAngles = {0f, 0f, 0f, 0f, 0f};
    [SerializeField] private float[] pinchBendAngles = {0f, 45f, 60f, 80f, 90f};
    [SerializeField] private float[] pointBendAngles = {0f, 0f, 0f, 80f, 90f};

    // 组件引用
    private UdpReceiver udpReceiver;
    private Animator handAnimator;

    // 状态变量
    private string currentGesture = "Neutral";
    private string currentState = "Neutral";
    private float currentConfidence = 0f;
    private Color targetColor;
    private Color currentColor;

    // 动画相关
    private Dictionary<string, float[]> gestureAnimations;
    private float[] currentBendAngles;
    private float[] targetBendAngles;

    // 特效相关
    private Coroutine colorTransitionCoroutine;
    private Coroutine gestureAnimationCoroutine;
    private Coroutine particleEffectCoroutine;

    void Start()
    {
        InitializeComponents();
        InitializeGestureAnimations();
        InitializeVisualEffects();
    }

    void OnDestroy()
    {
        // 清理协程
        StopAllCoroutines();
    }

    /// <summary>
    /// 初始化组件
    /// </summary>
    private void InitializeComponents()
    {
        // 获取UDP接收器
        udpReceiver = FindObjectOfType<UdpReceiver>();
        if (udpReceiver != null)
        {
            udpReceiver.OnGestureChanged += OnGestureChanged;
            udpReceiver.OnStateChanged += OnStateChanged;
            udpReceiver.OnDataReceived += OnDataReceived;
        }
        else
        {
            Debug.LogWarning("未找到UdpReceiver组件");
        }

        // 获取动画器
        handAnimator = GetComponent<Animator>();
        if (handAnimator == null)
        {
            Debug.LogWarning("未找到Animator组件");
        }

        // 初始化弯曲角度数组
        currentBendAngles = new float[5];
        targetBendAngles = new float[5];

        // 初始化颜色
        currentColor = neutralColor;
        targetColor = neutralColor;

        // 应用默认材质
        if (handRenderer != null && defaultMaterial != null)
        {
            handRenderer.material = defaultMaterial;
        }
    }

    /// <summary>
    /// 初始化手势动画配置
    /// </summary>
    private void InitializeGestureAnimations()
    {
        gestureAnimations = new Dictionary<string, float[]>
        {
            {"Fist", fistBendAngles},
            {"Open", openBendAngles},
            {"Pinch", pinchBendAngles},
            {"Point", pointBendAngles},
            {"Neutral", openBendAngles},
            {"Unknown", openBendAngles}
        };
    }

    /// <summary>
    /// 初始化视觉效果
    /// </summary>
    private void InitializeVisualEffects()
    {
        // 初始化粒子系统
        if (emotionParticles != null)
        {
            var emission = emotionParticles.emission;
            emission.rateOverTime = 0f; // 初始不发射
        }

        // 初始化手部光源
        if (handLight != null)
        {
            handLight.intensity = 0f;
        }

        // 初始化手指轨迹
        if (fingerTrails != null)
        {
            foreach (var trail in fingerTrails)
            {
                if (trail != null)
                {
                    trail.enabled = false;
                }
            }
        }
    }

    /// <summary>
    /// 手势变化事件处理
    /// </summary>
    private void OnGestureChanged(string gesture, float confidence)
    {
        if (currentGesture != gesture)
        {
            currentGesture = gesture;
            currentConfidence = confidence;

            // 触发手势动画
            StartGestureAnimation(gesture, confidence);

            Debug.Log($"手势变化: {gesture} (置信度: {confidence:F3})");
        }
    }

    /// <summary>
    /// 状态变化事件处理
    /// </summary>
    private void OnStateChanged(string state, float confidence)
    {
        if (currentState != state)
        {
            currentState = state;
            currentConfidence = confidence;

            // 触发状态颜色变化
            StartColorTransition(state, confidence);

            // 触发粒子效果
            StartParticleEffect(state, confidence);

            Debug.Log($"状态变化: {state} (置信度: {confidence:F3})");
        }
    }

    /// <summary>
    /// 数据接收事件处理
    /// </summary>
    private void OnDataReceived(UdpReceiver.EmotionData data)
    {
        // 更新实时数据
        currentGesture = data.gesture;
        currentState = data.state;
        currentConfidence = data.confidence;

        // 根据置信度调整效果强度
        UpdateEffectIntensity(data.confidence);
    }

    /// <summary>
    /// 开始手势动画
    /// </summary>
    private void StartGestureAnimation(string gesture, float confidence)
    {
        if (gestureAnimationCoroutine != null)
        {
            StopCoroutine(gestureAnimationCoroutine);
        }

        gestureAnimationCoroutine = StartCoroutine(AnimateGesture(gesture, confidence));
    }

    /// <summary>
    /// 手势动画协程
    /// </summary>
    private IEnumerator AnimateGesture(string gesture, float confidence)
    {
        // 获取目标弯曲角度
        if (gestureAnimations.TryGetValue(gesture, out float[] angles))
        {
            targetBendAngles = (float[])angles.Clone();
        }
        else
        {
            targetBendAngles = (float[])openBendAngles.Clone();
        }

        // 平滑过渡到目标角度
        float duration = 1.0f / transitionSpeed;
        float elapsedTime = 0f;
        float[] startAngles = (float[])currentBendAngles.Clone();

        while (elapsedTime < duration)
        {
            elapsedTime += Time.deltaTime;
            float t = elapsedTime / duration;

            // 应用平滑插值
            for (int i = 0; i < 5; i++)
            {
                currentBendAngles[i] = Mathf.Lerp(startAngles[i], targetBendAngles[i], t);

                // 应用到手指骨骼
                if (fingerBones != null && i < fingerBones.Length && fingerBones[i] != null)
                {
                    Vector3 localRotation = fingerBones[i].localRotation.eulerAngles;
                    localRotation.x = currentBendAngles[i];
                    fingerBones[i].localRotation = Quaternion.Euler(localRotation);
                }
            }

            yield return null;
        }

        // 确保最终角度正确
        for (int i = 0; i < 5; i++)
        {
            currentBendAngles[i] = targetBendAngles[i];

            if (fingerBones != null && i < fingerBones.Length && fingerBones[i] != null)
            {
                Vector3 localRotation = fingerBones[i].localRotation.eulerAngles;
                localRotation.x = currentBendAngles[i];
                fingerBones[i].localRotation = Quaternion.Euler(localRotation);
            }
        }
    }

    /// <summary>
    /// 开始颜色过渡
    /// </summary>
    private void StartColorTransition(string state, float confidence)
    {
        if (colorTransitionCoroutine != null)
        {
            StopCoroutine(colorTransitionCoroutine);
        }

        colorTransitionCoroutine = StartCoroutine(TransitionColor(state, confidence));
    }

    /// <summary>
    /// 颜色过渡协程
    /// </summary>
    private IEnumerator TransitionColor(string state, float confidence)
    {
        // 获取目标颜色
        switch (state)
        {
            case "Relaxed":
                targetColor = relaxedColor;
                break;
            case "Focused":
                targetColor = focusedColor;
                break;
            case "Stressed":
                targetColor = stressedColor;
                break;
            case "Fatigued":
                targetColor = fatiguedColor;
                break;
            default:
                targetColor = neutralColor;
                break;
        }

        // 根据置信度调整颜色强度
        targetColor = Color.Lerp(neutralColor, targetColor, confidence);

        // 平滑颜色过渡
        float duration = 1.0f / transitionSpeed;
        float elapsedTime = 0f;
        Color startColor = currentColor;

        while (elapsedTime < duration)
        {
            elapsedTime += Time.deltaTime;
            float t = elapsedTime / duration;

            currentColor = Color.Lerp(startColor, targetColor, t);

            // 应用到手部材质
            if (handRenderer != null && handRenderer.material != null)
            {
                handRenderer.material.color = currentColor;

                // 设置发光效果
                if (handRenderer.material.HasProperty("_EmissionColor"))
                {
                    handRenderer.material.SetColor("_EmissionColor", currentColor * 0.3f);
                }
            }

            // 更新手部光源颜色
            if (handLight != null)
            {
                handLight.color = currentColor;
            }

            yield return null;
        }

        // 确保最终颜色正确
        currentColor = targetColor;

        if (handRenderer != null && handRenderer.material != null)
        {
            handRenderer.material.color = currentColor;

            if (handRenderer.material.HasProperty("_EmissionColor"))
            {
                handRenderer.material.SetColor("_EmissionColor", currentColor * 0.3f);
            }
        }

        if (handLight != null)
        {
            handLight.color = currentColor;
        }
    }

    /// <summary>
    /// 开始粒子效果
    /// </summary>
    private void StartParticleEffect(string state, float confidence)
    {
        if (particleEffectCoroutine != null)
        {
            StopCoroutine(particleEffectCoroutine);
        }

        particleEffectCoroutine = StartCoroutine(PlayParticleEffect(state, confidence));
    }

    /// <summary>
    /// 粒子效果协程
    /// </summary>
    private IEnumerator PlayParticleEffect(string state, float confidence)
    {
        if (emotionParticles == null)
            yield break;

        var emission = emotionParticles.emission;
        var main = emotionParticles.main;

        // 根据状态配置粒子效果
        switch (state)
        {
            case "Relaxed":
                main.startColor = relaxedColor;
                emission.rateOverTime = particleEmissionRate * confidence * 0.5f;
                break;
            case "Focused":
                main.startColor = focusedColor;
                emission.rateOverTime = particleEmissionRate * confidence * 0.7f;
                break;
            case "Stressed":
                main.startColor = stressedColor;
                emission.rateOverTime = particleEmissionRate * confidence * 1.2f;
                break;
            case "Fatigued":
                main.startColor = fatiguedColor;
                emission.rateOverTime = particleEmissionRate * confidence * 0.3f;
                break;
            default:
                emission.rateOverTime = 0f;
                break;
        }

        // 播放粒子效果
        emotionParticles.Play();

        // 持续一段时间后减少发射
        yield return new WaitForSeconds(2.0f);

        float fadeDuration = 1.0f;
        float elapsedTime = 0f;
        float startRate = emission.rateOverTime.constant;

        while (elapsedTime < fadeDuration)
        {
            elapsedTime += Time.deltaTime;
            float t = elapsedTime / fadeDuration;
            emission.rateOverTime = Mathf.Lerp(startRate, 0f, t);
            yield return null;
        }

        emission.rateOverTime = 0f;
    }

    /// <summary>
    /// 更新效果强度
    /// </summary>
    private void UpdateEffectIntensity(float confidence)
    {
        // 更新光源强度
        if (handLight != null)
        {
            handLight.intensity = confidence * lightIntensityMultiplier;
        }

        // 更新手指轨迹
        if (fingerTrails != null && confidence > 0.7f)
        {
            foreach (var trail in fingerTrails)
            {
                if (trail != null)
                {
                    trail.enabled = true;
                    trail.material.color = currentColor;
                }
            }
        }
        else if (fingerTrails != null)
        {
            foreach (var trail in fingerTrails)
            {
                if (trail != null)
                {
                    trail.enabled = false;
                }
            }
        }
    }

    /// <summary>
    /// 获取当前状态信息
    /// </summary>
    public string GetStatusInfo()
    {
        return $"手势: {currentGesture} | 状态: {currentState} | 置信度: {currentConfidence:F3}";
    }

    /// <summary>
    /// 重置可视化状态
    /// </summary>
    public void ResetVisualization()
    {
        currentGesture = "Neutral";
        currentState = "Neutral";
        currentConfidence = 0f;
        targetColor = neutralColor;

        // 重置颜色
        if (colorTransitionCoroutine != null)
        {
            StopCoroutine(colorTransitionCoroutine);
        }
        colorTransitionCoroutine = StartCoroutine(TransitionColor("Neutral", 0f));

        // 重置手势
        if (gestureAnimationCoroutine != null)
        {
            StopCoroutine(gestureAnimationCoroutine);
        }
        gestureAnimationCoroutine = StartCoroutine(AnimateGesture("Neutral", 0f));

        // 停止粒子效果
        if (emotionParticles != null)
        {
            emotionParticles.Stop();
        }

        // 重置光源
        if (handLight != null)
        {
            handLight.intensity = 0f;
        }

        // 重置轨迹
        if (fingerTrails != null)
        {
            foreach (var trail in fingerTrails)
            {
                if (trail != null)
                {
                    trail.enabled = false;
                }
            }
        }
    }

    void OnGUI()
    {
        // 显示状态信息
        GUILayout.BeginArea(new Rect(10, 10, 300, 100));
        GUILayout.Label($"手势: {currentGesture}");
        GUILayout.Label($"状态: {currentState}");
        GUILayout.Label($"置信度: {currentConfidence:F3}");

        if (udpReceiver != null)
        {
            GUILayout.Label($"UDP状态: {udpReceiver.GetStatusInfo()}");
        }

        GUILayout.EndArea();
    }
}