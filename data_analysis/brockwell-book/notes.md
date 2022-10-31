# 时间序列阅读笔记

## 引言

### 通用的时间序列建模方法
操作步骤：
- 画出主要feature的图，并查看
  - 是否有一个趋势(Trend)
  - 是否有周期的成分(Period)
  - 是否有突变(Sharp change)
  - 是否有异常值(Outlier)
- 移除趋势和周期项，获得一个接近静态的时间序列
- 建模拟合剩余的残差项
- 预测残差，并加上趋势和周期项，获得最终的预测值

### 静态模型
> **Def**: Weakly Stationary
> A time series ${X_t}$ is called weakly stationary, if 
> 1. $\mu_X(t)$ is independent of time $t$
> 2. $\gamma_X(t+h,t)$ is independent of time $t$ for each $h$, where $\gamma_X(r, s) = Cov(X_r, X_s)$

对于一个静态序列来说，$\gamma_X(h):=\gamma_X(0,h)=\gamma_X(t,t+h)$, 此时，我们称$\gamma_X(h)$ 为自协方差(auto covariance function). 

> **Def**: AutoCorrelation Function(自相关函数)
> $\rho_X(h) = \frac{\gamma_X(h)}{\gamma_X(0)}$

### 估计趋势和周期
假设一个时间序列由三个部分构成：
$$
X_t = m_t + s_t + Y_t
$$
其中$m_t, s_t$分别表示确定性的趋势项，周期项。$Y_t$剩余的一个带有随机性的静态序列。这节我们简单介绍一下一些简单的方法估计$m_t, s_t$

#### 只有趋势项的时候
如果我们的模型只有趋势项，如下
$$
X_t = m_t + Y_t
$$
那么常见的用来估计趋势项的方法：
- 移动平均平滑
  - 等权平均
  - 指数平均
- 过滤高频成分
  - 傅立叶变换
  - 其他的谱分析方法
- 参数化模型并拟合
  - 多项式拟合
  - 机器学习，深度学习方法
  
当然，我们也可以反其道而行，除了我们通过上述的方法，去获取一个预测趋势的模型，我们也可以试图消除趋势，直接获得静态序列$Y_t$。
消除趋势最常见的方法则是：
- 差分

> **Def**:Lag operator
> $$ BX_t = X_{t-1} $$
> 
> **Def**: Difference operator
> $$\nabla = 1 - B$$

因此如果$m_t$是一个$k$阶的多项式，那么$\nabla^k X_t$ 则足以消除趋势。

#### 同时有趋势项和周期项
上面介绍完了，在只有趋势项的是，我们如何去处理一个时间序列，这边我们考虑的模型更加完整一点，即趋势项和周期项同时存在。
假设周期为$d$, 一般我们可以遵从下面的步骤：
- 使用当前时间点附近$d$个时间的数据做移动平均，大概估计出一个$\hat{m}_t$
- 对于周期内的任意一个时间点$k, 1 \le k \le d$, 计算$\{x_{k+jd}-\hat{m}_{k+jd}\}, 1 \le k+jd \le n$的平均，计作$w_t$, 为了保证周期震荡总体围绕0点震荡，我们可以使用$\hat{s}_t = w_t - \frac{1}{d}\sum_{k=1}^d w_k$ 作为一个不错的周期项估计
- 现在我们获得了**去周期**的数据$d_t = x_t -\hat{s}_t$, 问题则变成了，上面只有趋势项的问题了，我们就可以上一小段列举的方法解决这个问题了。