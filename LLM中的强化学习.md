~~"Talk is cheap, show me the code." - Linus Torvalds~~

|概念|马里奥中的例子|符号表示|
|--|--|--|
|环境 Environment|游戏画面，游戏中的奖励，游戏何时结束等||	
|智能体 Agent|马里奥||
|状态 State、Observation|游戏当前的状态|$s_t$|
|动作 Action|马里奥做出的动作（左 | 右 | 下 | 跳）|$a_t$|
|奖励 Reward|马里奥做出一个动作以后，环境给出的奖励|$r_t$|
|轨迹 Trajectory|游戏中一连串状态和动作的序列|$\tau=\{s_0,a_0,s_1,a_1,\cdots,s_T\}$|
|策略函数 Policy|马里奥在当前状态下选择动作的概率分布|$\pi_\theta(\cdot\|s_t)$|
|回报 Return|在马里奥的一次完整轨迹中获得的奖励总和|$R(\tau)$|

## 1.	PG（Policy Gradient）策略梯度算法
来回顾一下期望的定义：
$$E(f(x))_{x\sim p}=\sum_n f(x)\cdot p(x)\approx\frac{1}{n}\sum^n_{i=1}f(x)$$
我们的目标是：训练一个Policy，使得Agent能够进行回报最大的轨迹
我们把$x$换成对轨迹$\tau$的回报值$R(\tau)$，即可得到下面这个表述：
$$E(R(\tau))_{\tau\sim p_\theta}=\sum_\tau R(\tau)p_\theta(\tau)$$
$$\begin{align} p_\theta(\tau)&=p(s_0)\pi_\theta(a_0|s_0)p(s_1|s_0,a_0)\pi_\theta(a_1|s_1)p(s_2|s_1,a_1)\pi_\theta(a_2|s_2)\cdots\notag\\ &=\pi_\theta(a_0|s_0)\pi_\theta(a_1|s_1)\pi_\theta(a_2|s_2)\cdots\notag\\ &\rightarrow \pi_\theta(\tau)\notag \end{align}$$
本来为了响应要更易于大家理解防止睡着或者跑路的号召，不打算推公式。但是强化学习算法的公式真不难（至少是从PG开始，从头到尾只需要围绕“期望”，没有其他多余的了），而且我觉得看公式其实更直观，所以还是会以边推公式边讲解的形式进行~
在实际情况下，得到所有的轨迹$\tau$是不可能的，所以还是采样$N$个轨迹进行估计：
$$E(R(\tau))_{\tau\sim \pi_\theta}\approx\frac{1}{N}\sum^N_{i=1}R(\tau_i)$$
这么做对吗，式子好像没啥问题，但是$\pi_\theta$不见了，那我更新策略时找谁更新去？应该先把$\pi_\theta$保留下来：
$$\begin{align} \nabla_\theta E(R(\tau))_{\tau\sim \pi_\theta}&=\nabla_\theta\sum_{\tau}R(\tau)\pi_\theta(\tau)\notag\\ &=\sum_{\tau}R(\tau)\nabla_\theta\pi_\theta(\tau)\notag\\ &=\sum_{\tau}R(\tau)\nabla_\theta \log\pi_\theta(\tau)\cdot\pi_\theta(\tau)\notag\qquad注：\nabla\log(f(x))=\frac{\nabla f(x)}{f(x)}\notag\\ &\approx\frac{1}{N}\sum^N_{i=1} R(\tau_i)\nabla_\theta\log \pi_\theta(\tau_i)\notag\\ &=\frac{1}{N}\sum^N_{i=1}R(\tau_i)\nabla_\theta\log\prod^{T_i}_{t=1}\pi_\theta(a_{i,t}|s_{i,t})\notag\\ &=\frac{1}{N}\sum^N_{i=1}R(\tau_i)\sum^{t_i}_{t=1}\nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})\notag\\ &=\frac{1}{N}\sum^N_{i=1}\sum^{T_i}_{t=1}R(\tau_i)\nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})\notag \end{align}$$
那么优化目标就很明确了，即如果$R(\tau_i)$是大于0的，那么就应该增加轨迹$\tau_i$中所有状态$s_{i,t}$下采取动作$a_{i,t}$的概率；反之如果$R(\tau_i)$是小于0的，那么就应该减小轨迹$\tau_i$中所有状态$s_{i,t}$下采取动作$a_{i,t}$的概率。
相信很多人听过PG算法通过梯度上升来更新策略，以最大化期望回报。但梯度上升其实就是没加负号，实现起来照样是加了符号的loss用梯度下降：
$$\mathcal{L}=-\frac{1}{N}\sum^N_{i=1}\sum^{T_i}_{t=1}R(\tau _i)\log\pi_\theta(a_{i,t}|s_{i,t})$$
自此我们得到了训练一个策略网络$\pi_\theta$的方法，以马里奥游戏为例，具体来说便是：
1.	在每个时间步$t$输入游戏画面$s_t$到$\pi_\theta$（比如一个简单的CNN），从输出概率$\pi_\theta(\cdot|s_t)$中采样动作$a_t$
2.	重复进行直到游戏结束，获得一个完整的轨迹$\tau$，计算回报$R(\tau)$
3.	得到n个轨迹后（1个batch），进行一次loss计算，backward，梯度更新。
目前存在什么问题？

- 目前一个轨迹$\tau$中每个时间步对于总回报的贡献是一致的都是$R(\tau)$。但是是否增大或者减小在时间步$t$下处于$s_t$进行动作$a_t$的概率，应该看做了这个动作之后到游戏结束累计的回报，而不应该是整个轨迹$\tau$累计的回报。更进一步地，这个回报应该随着距离当前时间步$t$越远而逐步衰减。
  由此定义$R(\tau_i)\rightarrow R(\tau_i,t)$：
  $$R(\tau_i,t)=\sum^{T_i}_{t'=t}\gamma^{t'-t}r_{i,t'}$$
  这里便用的是蒙特卡洛方法。蒙特卡洛方法相信大家都有了解，在这里就是通过不断完整的采样轨迹来估算回报。它有个特点就是基于观测到的样本的完整轨迹。
- 对于好的局势来说，可能马里奥不管怎么行动，得到的奖励都是正的；反之对于坏的局势来说，不管怎么行动，得到的奖励都是负的。但是这会导致训练很慢，因为会增大/减小所有动作的概率。更加合适的做法是增大相对好的动作的概率而减小相对差的动作的概率（这里mark一下~），为此可以引入一个baseline：
  $$\begin{align} \mathcal{L}=-\frac{1}{N}\sum^N_{i=1}\sum^{T_i}_{t=1}\left(R(\tau_i,t)-B(s_{i,t})\right)\log\pi_\theta(a_{i,t}|s_{i,t}) \end{align}$$
  但是这个$B(s_{i,t})$的值怎么得到呢？PG原始论文里使用的是与策略无关的简单baseline（如状态相关的平均值、随时间衰减的启发式值，基于统计量），后来...马上讲到。
  至此！就是1992年原始PG论文里包含的算法以及改进了。从下面开始都是以后的研究在此基础上进行的改进，在这里说明一下~
- 之前的简单baseline$B(s_{i,t})$疑似有点太简单了，效果很有限。能不能有个方法估计一下在状态$s_t$时期望的回报？
  由此定义$V(s_t)$，状态价值函数
- $R(\tau_i,t)$每次都是一次完整采样后得到的（蒙特卡洛方法），虽然偏差小，但是方差很大，训练不稳定。能不能有个方法估计一下在状态$s_t$时进行动作$a_t$的期望回报？
  由此定义$Q(s_t, a_t)$，动作价值函数
  $$Q(s_t,a_t)=r_t+\gamma\cdot V(s_{t+1})$$
  这里便用的是时序差分方法（一步采样）。可能有人这里就有疑问了，乍一看这新定义的动作价值函数不就是蒙特卡洛方法的一步罢了吗？但是蒙特卡洛方法用的全都是真实的奖励，而时序差分用的是价值函数估计。对比蒙特卡洛方法，降低了方差，但是引入了偏差。

至此我们在公式$(1)$中拿到的$R(\tau,t)-B(s_{t})$便可以转化为一个全新的定义：
$$A(s_{t},a_{t})=Q(s_t,a_t)-V(s_t)=r_t+\gamma\cdot V(s_{t+1})-V(s_t)$$
这就是优势函数（Advantage Function）
蒙特卡洛方法无偏但是方差大，时序差分方法方差小但是有偏。能不能结合一下？
我们不妨先看看多步采样时序差分算法：
$$\begin{align} A(s_t,a_t)^{(1)}&=r_t+\gamma\cdot V(s_{t+1})-V(s_t)\notag\\ A(s_t,a_t)^{(2)}&=r_t+\gamma\cdot r_{t+1}+\gamma^2\cdot V(s_{t+2})-V(s_t)\notag\\ A(s_t,a_t)^{(3)}&=r_t+\gamma\cdot r_{t+1}+\gamma^2\cdot r_{t+2}+\gamma^3\cdot V(s_{t+3})-V(s_t)\notag\\ \cdots\notag\\ A(s_t,a_t)^{(T-t+1)}&=r_t+\gamma\cdot r_{t+1}+\gamma^2\cdot r_{t+2}+\cdots+\gamma^T\cdot r_{T}-V(s_t)\notag \end{align}$$
可以看出，当步数为$T-t+1$即所有后续步，就变为了蒙特卡洛方法。那么结合两种方法的思路就有了，即给第1步、第2步等等后续所有步采样分配不同的权重进行加和：
$$\begin{align} A^{GAE}(s_t,a_t)&=(1-\lambda)(A(s_t,a_t)^{(1)}+\lambda A(s_t,a_t)^{(2)}+\lambda^2 A(s_t,a+t)^{(3)}\cdots)\notag\\ &\quad\cdots\notag\\ &=\sum^{T-t}_{l=0}(\gamma\lambda)^l(r_{t+l}+\gamma\cdot V(s_{t+l+1})-V(s_{t+l}))\notag \end{align}$$
这就是广义优势估计（Generalized Advatnage Estimation, GAE）。OK，我们已经成功得到了非常现代的PG算法！我们回到loss函数看看我们具体要怎么进行训练：
$$\begin{align} \mathcal{L}=-\frac{1}{N}\sum^N_{i=1}\sum^{T_i}_{t=1}A^{GAE}(s_{i,t},a_{i,t})\log\pi_\theta(a_{i,t}|s_{i,t})\notag \end{align}$$
$\pi_\theta$是我们要训练的策略网络，回到马里奥游戏中，就是一个接收画面输入，输出层是nn.Linear(hidden_size, 4)和softmax的神经网络Actor Model。$$A^{GAE}(s_t,a_t)$$中的$V(s_{i,t})$用另一个神经网络进行拟合，我们称之为Critic Model，其输入与Actor Model一致，输出层为nn.Linear(hidden_size, 1)，输出一个标量，表示当前状态的价值估计。Actor Model用上面的Loss进行优化，Critic Model用MSE Loss。这就是Actor-Critic + GAE算法结构，同时对两个网络进行训练。

## 2.	PPO（Proximal Policy Optimzation）近端策略优化算法
看标题像是从PG算法直接到了PPO，其实在上一节中其实包含原始PG算法、Actor-Critic算法，Actor-Critic + GAE算法（现代PG算法）。但我没起那么多小标题是为了让讲解更顺畅有一步一步优化的逻辑感。所以这节肯定也不是直接到了PPO，下面正式开始。
首先从强化学习中一个没什么意义但又很多人喜欢纠结的概念开始：On-policy和Off-policy
 
回到模型中来讲，on-policy就是用当前的策略模型用自己采样得到的数据去计算loss进行训练；off-policy当然就和这相反，当前的策略模型不是用自己采样得到的数据去计算loss进行训练。
因此我们可以发现，上面讲的所有方法都是on-policy的，但为什么会引入off-policy呢？
数据采样的成本太高啦！训练效率太低。要一直不断 更新->采样->更新->采样->...
设想用off-policy，就可以反复使用历史采样到的大量数据进行训练了，效率大大提升~
但肯定不能说用就用，我们回到最开始的期望。
很显然我们要把$\tau\sim\pi_\theta$变成$\tau\sim\pi_{\theta'}$
我们推导一下重要性采样：
$$\begin{align} E(f(x))_{x\sim p}&=\sum_xf(x)\cdot p(x)\notag\\ &=\sum_xf(x)\cdot p(x)\cdot\frac{q(x)}{q(x)}\notag\\ &=\sum_xf(x)\frac{p(x)}{q(x)}\cdot q(x)\notag\\ &=E(f(x)\frac{p(x)}{q(x)})_{x\sim q}\notag \end{align}$$
所以loss直接可以变成：
$$\begin{align} \mathcal{L}=-\frac{1}{N}\sum^N_{i=1}\sum^{T_i}_{t=1}A^{GAE}(s_{i,t},a_{i,t})\frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\theta'}(a_{i,t}|s_{i,t})}\notag \end{align}$$
看起来很美好，但是有个限制，就是$\pi_{\theta'}$和$\pi_\theta$分布不能相差太大。因为通过重要性采样，期望虽然不变，但是方差会变，可以简单看一眼：
$$Var(f(x))_{x\sim p}=E(f(x)^2)_{x\sim p}-E(f(x))^2_{x\sim p}$$
$$\begin{align} Var(f(x)\frac{p(x)}{q(x)})_{x\sim q}&=E(f(x)^2\frac{p(x)^2}{q(x)^2})_{x\sim q}-E(f(x)\frac{p(x)}{q(x)})^2_{x\sim q}\notag\\ &=E(f(x)^2\frac{p(x)}{q(x)})_{x\sim p}-E(f(x))^2_{x\sim p}\notag \end{align}$$
所以很容易就能想到，为了不让两者相差太大，用KL散度进行约束不就ok了，于是就有了TRPO（Trust Region Policy Optimization）置信域策略优化算法，其将KL散度约束加入到Loss之中：
$$\begin{align} \mathcal{L}_{TRPO}=-\frac{1}{N}\sum^N_{i=1}\sum^{T_i}_{t=1}A^{GAE}(s_{i,t},a_{i,t})\frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\theta'}(a_{i,t}|s_{i,t})}\quad s.t.\quad KL(\pi_\theta,\pi_{\theta'})<\sigma\notag \end{align}$$
但是带不等式约束的优化问题，在实际使用时几乎是无法实现的，因此原始PPO算法将约束直接增加到目标函数之中：
$$\begin{align} \mathcal{L}_{PPO}=-\frac{1}{N}\sum^N_{i=1}\sum^{T_i}_{t=1}A^{GAE}(s_{i,t},a_{i,t})\frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\theta'}(a_{i,t}|s_{i,t})}+\beta KL(\pi_\theta,\pi_{\theta'})\notag \end{align}$$
但是这种方法在实际实现中计算量也有点大了，并且OpenAI的PPO论文原文里提到，惩罚系数$\beta$的选择非常困难，在不同任务甚至同一任务的不同阶段，最优的$\beta$值都可能发生变化。所以PPO-Clip直接将概率比值限定在一定范围内：
$$\begin{align} \mathcal{L}_{PPO-Clip}=-\frac{1}{N}\sum^N_{i=1}\sum^{T_i}_{t=1}\min\left(\frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\theta'}(a_{i,t}|s_{i,t})}A^{GAE}(s_{i,t},a_{i,t}),clip\left(\frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\theta'}(a_{i,t}|s_{i,t})},1-\epsilon,1+\epsilon\right)A^{GAE}(s_{i,t},a_{i,t})\right)\notag \end{align}$$
至此！我们得到了OpenAI在RLHF中使用的PPO算法（的大致形态）！
现在接入LLM背景！让我们结合模型来进行接下来的理解。
首先明确之前的概念放到LLM领域中是什么。我们知道LLM其实就是接收prompt输出response，response由一些列输出tokens $\{x_1,x_2,\cdots,c_T\}$组成。那么初始状态$s_0$就是prompt，动作$a_0$就是$x_1$，那么状态$s_1$就是$\text{prompt}+x_1$。此外，即时奖励$r_t$可没什么好的人为设定标准，因此肯定需要一个模型去给定。
 
我们来数一数换到LLM背景里，实现这样一个PPO算法需要几个模型。
1.	$\pi_\theta$：首先肯定是我们最主要的待优化的策略模型Policy Model（之前叫Actor Model）
2.	$r_\phi$：$A^{GAE}$中计算即时奖励的奖励模型Reward Model
3.	$V_\psi$：$A^{GAE}$中计算预估总回报的价值模型Value(Critic) Model
4.	$\pi_{\theta_{old}}$：用来进行实际采样的旧策略模型Old Policy Model
5.	$\pi_{\theta_{ref}}$：用来计算KL惩罚的参考模型Reference Model
这些模型是什么？
- Policy Model和Reference Model都是Base Model经过SFT训练得到的完全相同的两个模型副本，只不过后者冻结了。假设是两个175B的GPT-3。
- Reward Model则是通过pair-wise loss训练得到的，专门对一个完整的序列输出奖励得分的模型，最后一层是nn.Linear(hidden_size, 1)，训练过程冻结。可以使用一个小很多的模型比如6B的GPT-3，把最后一层换掉，但通常都是一个和Policy Model参数量一样的模型。
简单看一眼Reward Model训练时的Loss函数：
$$\mathcal{L}_r=-E_{(x,y_w,y_l)\sim \mathcal{D}}\left[\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)\right]$$
其中$\sigma$是Sigmoid函数，可以将$r_\phi(x,y_w)-r_\phi(x,y_l)$转化为一个概率。
- Value Model则是直接由训练好的Reward Model初始化来，最后一层同样是nn.Linear(hidden_size, 1)，参与训练。
ok，那Old Policy Model是个什么东西？？为什么之前讲的on-policy和off-policy的概念以及重要性采样好像完全没有体现？我们先从上图来一一对应公式里的各项~梳理一下图中的流程：
1.	用$\pi_{\theta}$（$\pi_{\theta_{old}}$）采样$N$个输出
2.	使用$r_{\phi}$计算即时奖励
3.	使用$\pi_{\theta_{ref}}$计算KL惩罚项
4.	使用$V_\psi$计算价值
5.	使用2.3.4步计算的值计算得到$A^{GAE}$
6.	计算loss更新$\pi_\theta$与$V_\psi$
7.	重复3-6步PPO-step次
看第7步，对于一次采样，模型更新了PPO-step次。这里就体现出了Old Policy Model的用处，相当于提供了一个动态锚点，用于重要性权重计算。
乍一看，好像和我们刚才拿到的PPO-Clip有哪些地方好像对不上？确实是的，一方面是因为放到LLM领域中会做一些适应性的调整，另一方面是OpenAI确实做了一些小变化。
先看放到LLM领域中做的调整：
- Reward Model不再对每个输出token输出一个即时奖励值了，而是只有最后一个token即获得完整序列后有reward，之前的reward为0。
- 我们在这里叫之前的$\theta'$为Old Policy Model而不是一个其他的模型，说明与以往用一个其他的网络来采样的不同，LLM中使用Policy Model在完整进行PPO-step次前的自己的副本（然而实际实现上并不存在这样一个副本拖累显存，因为只需要保存log_probs就行）。但是这里就有个纠结点了，这到底是on-policy还是off-policy？结论是on-policy，因为相差不大。
再来看看OpenAI做的小变化：
- 又把KL惩罚加回来了，加到了计算每个输出token的即时奖励中，而不是加在最终的Loss之中。
- 其实变版PPO-ptx还加入了预训练目标，这个好理解，就是在loss中加入了预训练loss。（但是主流的方法并不常用这个）

以上，便是现在主流的PPO算法的实现了！当年赫赫有名的RLHF其实拆开来讲，from Human FeedBack就是让SFT模型对prompt输出K条回答，让人工去排序这些回答，从而可以得到$C_2^K$组pair，使用pair-wise loss训练Reward Model；RL就是PPO。

## 3. DPO（Direct Preference Optimization）直接偏好优化算法
PPO一下要4个模型！而且一堆超参数，有没有什么办法简化一些？
一个直观的想法就是把训练Reward Model时那种偏好数据拿来，设计一种Loss直接训练Policy Model不就行了？不训练Reward Model、不做RL采样优化，直接用人类偏好排序的数据，最大化Policy输出人类偏好样本的概率，同时抑制不偏好的样本。为什么听起来有点奇怪，因为DPO就不是强化学习:)
我们直接来看一眼Loss函数就差不多懂了：
$$\mathcal{L}_{DPO}=-E_{(x,y_w,y_l)\sim \mathcal{D}}\left[\log\sigma\left(\beta\frac{\pi_\theta(y_w|x)}{\pi_{\theta_{ref}}(y_w|x)}-\beta\frac{\pi_\theta(y_l|x)}{\pi_{\theta_{ref}}(y_l|x)}\right)\right]$$
可以看出和训练Reward Model时的Loss有异曲同工之妙，这里还隐式的引入了KL约束。

所以DPO比PPO更好吗？感觉看起来应该不会让模型效果变得更好。大概有几下几个原因：
- DPO完全使用的是离线数据，而不是PPO这种在线采样数据，非常依赖于数据规模。
- 缺少探索，在复杂任务比如代码、数学上，PPO 通过采样-更新-再采样能渐进式发现改写策略，而 DPO 只能记住哪段输出被标成好。
- ...

要说的话还能说出一些原因，但总结下来根本就是依赖于数据。Qwen系列就用的DPO，因为他们的数据数量已经足够多、质量足够好了。
在一般情况下来看，DPO相比PPO节省太多人力、算力资源，适合用于需要快速优化偏好的场景。
一般训练过程可以参考：预训练 -> SFT -> 拒绝采样 -> DPO -> PPO 后两者看情况可翻转
“我是不想PPO吗，我是用不起。”
## 4.	GRPO（Group Relative Policy Optimization）群体相对策略优化算法
DPO的适用范围有限，且在数学等复杂问题上能力弱，有没有没PPO要求那么高，但是对这类问题求解效果还很好的方法？DeepSeek-Math中提出的GRPO算法表示我可以。
从PPO的问题出发：
- 在LLM领域中，Reward Model不是对每个$(s_t,a_t)$（输出token）输出一个奖励分数，而是在整个序列生成完毕后给最后一个token奖励。这导致训练Value Model为每个token都生成一个期望奖励变得困难。
- 模型太多啦，显存炸了、计算消耗太大了。
我们知道Value Model就是来充当状态价值函数的计算基准的，那我们可以用同一个问题产生的多个采样输出的平均奖励作为基准，则Loss函数就理所当然的变成了：
$$\mathcal{L}_{GRPO}=-\frac{1}{G}\sum^G_{i=1}\frac{1}{|o_i|}\sum^{|o_i|}_{t=1}\left(\min\left(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{ref}}(o_{i,t}|q,o_{i,<t})}\hat A_{i,t},clip\left(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{ref}}(o_{i,t}|q,o_{i,<t})},1-\epsilon,1+\epsilon\right)\hat A_{i,t}\right)-\beta KL(\pi_\theta||\pi_{\theta_{ref}})\right)$$
$$\hat A_{i,t}=\frac{r_i-\text{mean}(r)}{\text{std}(r)}$$
$$KL(\pi_\theta||\pi_{\theta_{ref}})=\frac{\pi_{\theta_{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})}-\log\frac{\pi_{\theta_{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})}-1$$
这里使用的是一个对KL散度的K3无偏估计器，并不是传统的KL散度，看下就好了。
 
我们发现这里$\hat A_{i,t}$等式的右边只有$i$，和$t$无关，说明在这里分配给每个token的优势值是一样的。但这是DeepSeek-Math中用的第一种策略叫结果监督。
还有一种是过程监督，简单看一眼公式：
$$\hat r^\text{index(j)}=\frac{r_i^{\text{index}(j)}-\text{mean}(R)}{\text{std}(R)}$$
$$\hat A_{i,t}=\sum_{\text{index}(j)\geq t}\hat r^{\text{index}(j)}_i$$
简单来说就是奖励模型会输出所有输出tokens上的奖励值，然后做归一化。每个token上的优势值就是后续所有tokens的归一化奖励值总和。

但好像我在做模板生成这个任务时没用到Reward Model吖？
DeepSeek-R1中进一步优化了GRPO算法，直接把Reward Model从模型预测变成了简单的规则奖励了。又进一步省了不少显存的同时，很大程度上避免了Reward Hacking的问题。ps：用的是结果监督。
比如原始DeepSeek-R1用了简单的两个规则：
- 要求模型将思考过程放在<think>和</think>两个special tokens之间。
- 直接评估输出答案是否正确。例如，对于结果确定的数学问题，让模型在输出中把答案包在\box 里以便基于规则验证正确性。对于代码问题，可以使用通过执行测试用例来判断对不对。
后续HuggingFace对其复现的工作Open-R1中设计加入了更多的规则，如：
- 鼓励模型进行更多步骤的推理，比如统计思考过程中“1. 2. 3. ”、“首先 其次 然后”等出现的次数。
- 输出长度奖励，鼓励答案正确的输出长度更短，抑制答案错误的输出长度太长。
- ...

公式疑似太多了。讲到这里，来讲一下我最终在模板生成任务中的pipeline吧，以及存在的问题。
从两个尝试方向来展开，一种是不加入CoT，一种是加入CoT。

- 不加入CoT：
    - 构造输出结果中直接是可解析模板的SFT数据
    - 使用Qwen2.5-14B-Instruct做SFT训练
    - 构造GRPO训练数据，设计奖励函数
    - 做GRPO训练
- 加入CoT
    - 调用DeepSeek-671B模型生成包含CoT的SFT数据
    - 使用DeepSeek-R1-Distil-Qwen-14B做SFT训练
    - 构造GRPO训练数据，设计奖励函数
    - 做GRPO训练

使用了几种奖励函数：
- 格式奖励，用Pydantic验证输出的格式是否正确
- 素材使用合规奖励
- 透明度相差小的layouts之间不应有重叠，layout不应该超出分辨率
- ...

这一套流程下来最主要的问题有以下几点：
- 用来做SFT的高质量数据太少了，这有部分原因也来自本身任务设计的问题
- GRPO训练周期相对较长，但是卡资源不够稳定
- 任务所需的生成序列较长，模型本身很难以通过不断采样来优化得知类似layouts重叠涉及到的相关tokens

OK我们回来再继续看看字节提出的DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）对GRPO做了哪些优化。
- 移除了KL惩罚：在训练长思维链推理模型时，模型分布可能与初始模型显著偏离，因此这种限制不是必要的。这下连Refenrece Model都不用了，显存压力更小了~
- 提高裁剪上限：考虑$\pi_{\theta_{old}}(o_i|q)=0.01和0.9$，则更新后的最大概率$\pi_\theta(o_i|q)$为0.012和1.08，这就表示对于概率较高的token受到的约束较少，对于低概率token，要实现概率的显著增加要困难得多。
- 动态采样，对数据进行过采样，过滤掉准确率为0和1的prompts+responses：如果在一个group里的输出的Reward都是1，那么这个group的所有样本的优势值都是0，没梯度更新了。
- Token-Level Loss：GRPO是Sample-Level的Loss，其中对Loss的计算是首先在每个样本内按token计算平均损失，然后汇总样本间的损失。这会导致较长response中每个token对整体损失的贡献不成比例的较低。这可能导致这两个问题：1. 对于高质量的长样本，可能阻碍模型学习其中的推理相关模式。2. 过长的样本表现出低质量模式，如胡言乱语和重复词。
这里对比一下Loss就一目了然了：
$$\mathcal{L}_{GRPO}=\red{-\frac{1}{G}\sum^G_{i=1}\frac{1}{|o_i|}\sum^{|o_i|}_{t=1}}\left(\min\left(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{ref}}(o_{i,t}|q,o_{i,<t})}\hat A_{i,t},clip\left(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{ref}}(o_{i,t}|q,o_{i,<t})},1-\epsilon,1+\epsilon\right)\hat A_{i,t}\right)-\beta KL(\pi_\theta||\pi_{\theta_{ref}})\right)$$
$$\mathcal{L}_{DAPO}=\red{-\frac{1}{\sum_{i=1}^G|o_i|}\sum^G_{i=1}\sum^{|o_i|}_{t=1}}\left(\min\left(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{ref}}(o_{i,t}|q,o_{i,<t})}\hat A_{i,t},clip\left(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{ref}}(o_{i,t}|q,o_{i,<t})},1-\epsilon_{low},1+\epsilon_{high}\right)\hat A_{i,t}\right)\right)$$

与DAPO同时期出现的还有Dr. GRPO（Done Right）。直接看眼公式吧先：
$$\mathcal{L}_{DAPO}=\red{-\frac{1}{G}\sum^G_{i=1}\sum^{|o_i|}_{t=1}}\left(\min\left(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{ref}}(o_{i,t}|q,o_{i,<t})}\hat A_{i,t},clip\left(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{ref}}(o_{i,t}|q,o_{i,<t})},1-\epsilon,1+\epsilon\right)\hat A_{i,t}\right)\right)$$
$$\hat A_{i,t}=\red{r_i-\text{mean}(r)}$$
- 除数$|o_i|$直接没了：$\mathcal{L}_{GRPO}$和$\mathcal{L}_{DAPO}$中都有response-level长度偏见的问题——当优势值大于0时，这个除数会导致Policy Model更偏爱简短的正确回答；当优势值小于0时，这个除数会导致Policy Model更偏爱更长的错误回答。所以直接扔了~
- 优势值的计算去除了标准差这个除数：存在question-level难度偏见的问题——标准差小的group会得到更高的优势值从而梯度值更高，标准差小的group会得到更低的优势值从而梯度值更低。也就是说，对于简单的问题，采样的回答中得到的标准差一般较小，但反而梯度值大；对于较难的问题，采样的回答中得到的标准差较大，但反而梯度值小。

OK差不多到此完结！
最后贴一个我在知乎上看到的非常真实的语录：
 


