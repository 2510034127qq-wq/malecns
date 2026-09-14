完整阅读并严格遵守项目根目录 ~/malecns/AGENTS.md。

当前环境、依赖、Arbor CUDA/MPI、MaleCNS v1.0 全量数据、全部 SWC skeleton、逐突触表、MuJoCo 等均已准备完成。不要重新安装环境、重新下载数据，也不要把项目重构成缩水版。

现在直接自主实现最终目标：

【最终目标】

构建一个真正闭环的、可运行和可观察的数字果蝇沙箱：

3D 环境
  ↓
复眼视觉 / 嗅觉 / 触觉 / 本体感觉
  ↓
完整 MaleCNS
  ↓
DN / VNC / motor pathways
  ↓
六腿 + 双翅 + 头部的 MuJoCo 物理身体
  ↓
身体在环境中运动
  ↓
环境与身体状态重新产生感觉输入
  ↺

行为必须来自这个闭环。

禁止用类似：

if wall_detected:
    turn_left()

if food_on_right:
    move_right()

这种高级行为脚本控制果蝇。

环境可以提供物理事实、光、气味浓度、碰撞、关节状态等，但最终采取什么动作必须来自 CNS → motor 输出。

==================================================
一、完整 MaleCNS
==================================================

使用已经下载的全部 MaleCNS v1.0 数据：

- 全部 neuron annotations
- 全部 native SWC skeletons
- full connectome
- syn-points
- syn-partners
- neuron neurotransmitter predictions
- per-T-bar neurotransmitter probabilities
- body statistics
- 可利用的 ROI / neuropil / cell-type 信息

保持原始数据不可变。

MaleCNS 原始 SWC 和对应原生 synapse 坐标是 8 nm 单位。
在 ingestion boundary 统一转换：

1 unit = 8 nm = 0.008 µm

morphology 与 synapse XYZ 必须使用完全相同的坐标变换。

使用 Arbor cable_cell 构建 morphologically detailed multi-compartment neurons。

完整保留：

- 神经元
- morphology topology
- synaptic partner relation
- individual synapse locations
- available neurotransmitter information

允许 Arbor 做合理的 control-volume discretization。
不允许最终模型退化成 point-neuron connectome 或只保留 neuron-pair aggregate weight。

小样本只允许用于局部 unit test / debugging，不允许作为项目成果或架构目标。

==================================================
二、感觉系统
==================================================

果蝇不能直接读取“物体坐标”“食物在哪里”等高级环境真值来决定行为。

1. Vision

在果蝇头部建立左右复眼视角。

由 3D 环境从果蝇实际姿态渲染视觉输入，再建立 compound-eye sampling，把视觉信号编码到 MaleCNS 中有依据的视觉/photoreceptor 输入神经元。

数据流：

MuJoCo world
→ fly head pose
→ left/right eye render
→ compound-eye sampling
→ photoreceptor / visual sensory neurons
→ MaleCNS

视觉应随果蝇自身位置、朝向、头部姿态实时变化。

2. Olfaction

环境中允许存在 odor sources，例如食物。

建立空间浓度场 C(x,t)。

优先采用有物理意义且计算成本合理的模型，可包含：

- 距离衰减
- 扩散
- 风向
- 必要时的简化 plume

左右 antenna 在各自真实空间位置采样浓度，再编码到有依据的 olfactory receptor neurons。

CNS 不允许直接知道 odor source 的坐标。

3. Mechanosensation

从 MuJoCo 获取真实：

- leg-ground contact
- body collision
- contact force
- 其他有依据的 mechanical state

映射到相应 mechanosensory channels。

4. Proprioception

从 MuJoCo 读取：

- joint angle
- joint velocity
- body pose
- limb state
- actuator / contact state

映射到 MaleCNS/VNC 中有依据的 proprioceptive inputs。

最终必须形成：

motor
→ body
→ proprioception / contact
→ CNS

闭环。

==================================================
三、身体
==================================================

用 MuJoCo 创建果蝇物理身体。

至少包括：

- thorax/body
- head
- abdomen（如果建模合理）
- six articulated legs
- left/right wings
- relevant joints
- collision geometry
- mass/inertia
- ground contact
- friction
- body pose
- joint position/velocity

运动输出必须由 MaleCNS 中可解释的：

- descending neurons
- VNC motor pathways
- motor neuron / motor pool activity

驱动身体 actuator / force / torque。

需要建立明确、可检查的 neural-output → body-actuation mapping。

任何缺失的真实肌肉、生理、空气动力学参数都要：
1. 查可靠果蝇文献/成熟模型；
2. 记录来源；
3. 无可靠精确值时采用明确标注的合理假设；
4. 做成配置项；
5. 不允许把假设写成“真实值”。

飞行也属于最终目标。
如 MuJoCo 本身不能直接提供所需昆虫空气动力，则实现明确、可解释并有文献依据的 wing aerodynamic force model，把力/力矩施加到 MuJoCo body。
不要因为飞行复杂就从最终目标删除双翅飞行能力。

==================================================
四、3D 环境
==================================================

首先实现一个可以自由运行的标准沙箱环境。

至少能够包含：

- ground
- walls
- obstacles
- lights
- odor/food sources
- arbitrary simple geometry

并支持生成标准行为实验场景：

1. phototaxis：明暗选择
2. odor choice / Y-maze
3. obstacle avoidance
4. looming visual stimulus

这些场景只是给果蝇提供 stimulus。
禁止直接编码“正确反应”。

同时保留扩展为普通房间场景的能力：

- table
- cup
- fruit
- plants
- walls
- window
- lights
- imported meshes

环境资产层不要和神经模拟核心强耦合。

==================================================
五、最终可视化
==================================================

做成一个统一可运行的数字果蝇程序/工作台。

主界面至少同时提供：

A. MuJoCo 3D World
- 显示果蝇身体
- 环境
- 实时运动
- 可自由观察

B. MaleCNS 3D
- 使用真实 morphology
- 显示 brain / optic lobes / VNC
- neural activity overlay
- spike / Vm / activity 可视化
- 支持按 cell type / neuropil / activity 筛选

C. Left eye / Right eye
- 实时显示果蝇自己实际接收到的视觉输入
- 最好同时提供 compound-eye sampled representation

D. Sensory / motor telemetry
例如：
- visual input activity
- olfactory activity
- mechanosensory activity
- proprioception
- DN activity
- wing L/R drive
- six-leg motor activity
- body velocity/orientation

需要支持：

- pause
- resume
- simulation speed
- neuron selection
- neuron search by body ID
- inspect one neuron
- inspect its morphology
- inspect its incoming/outgoing synapses
- inspect current neural state
- inspect neurotransmitter prediction
- locate selected synapse in 3D
- follow activity from sensory pathway toward motor pathway

单神经元视图应尽可能显示：

Body ID
cell type
neuropil/ROI
NT
Vm / compartment state
input synapse count
output synapse count
selected synapse XYZ
pre/post partner

Viewer 使用 LOD。
远景无需逐帧绘制 1 亿级 synapses；
近景选择神经元时再显示真实 individual synapses。

显示层可以抽样/LOD，但 simulation/data model 不允许因此丢失 synapses。

==================================================
六、架构
==================================================

保持最小充分架构，避免过度工程化。

建议职责分离：

data/
morphology/
synapses/
parameters/
simulation/
sensory/
motor/
body/
environment/
viewer/
experiments/
tests/

Arbor 负责神经模拟。
MuJoCo 负责 body/environment physics。
Viewer 与 simulation 解耦。
必要时使用现有 pyzmq/msgpack 做低开销状态传输，但如果单进程/共享内存更简单，就优先简单方案。

不要引入数据库、微服务、Web 后端等没有实际需要的东西。

大数据读取：
- Arrow memory map
- chunking
- compact indexes
- 必要缓存

禁止无理由把 10~20 GB 数据完整复制两三份。

==================================================
七、科研真实性
==================================================

任何无法从 MaleCNS 本身获得的参数，例如：

- Cm
- Ra
- leak
- ion channels
- synaptic kinetics
- conductance
- transmission delay
- receptor dynamics
- sensory transfer functions
- muscle properties
- wing aerodynamics

都需要查可靠文献/成熟果蝇模型。

在项目里建立清晰的 provenance。

每个参数至少区分：

MEASURED
LITERATURE_DERIVED
INFERRED
ASSUMED

不要因为真实参数不完整而阻塞整个项目。
存在合理文献范围时采用有依据的默认值并做成配置。

但不允许凭空编造数据。

==================================================
八、验证
==================================================

Agent 自主开发并持续验证。

必须记录并核对：

- source neuron count
- skeleton count
- source synapse count
- partner count
- mapped synapse count
- unmapped synapse count
- ambiguous mapping count
- Arbor cell count
- Arbor connection/synapse count
- dropped records（必须为显式、可解释）
- RAM
- VRAM
- preprocessing time
- network construction time
- biological simulation time / wall-clock time

最终必须证明：

完整 MaleCNS network 已实际实例化；
simulation time 能实际向前推进；
感觉输入能进入 CNS；
CNS motor output 能影响 MuJoCo body；
body movement 能改变下一时刻 sensory input。

这才算真正闭环。

==================================================
九、工作方式
==================================================

你是该项目的主要开发 Agent。

自主完成工程，不要每完成一个小步骤就停下来问我。

先检查当前仓库、环境和完整数据的真实状态，然后开始实施。

遇到普通：
- Python error
- C++/CUDA error
- Arbor API error
- MuJoCo error
- 数据格式问题
- 性能问题
- viewer 问题

自行定位并修复。

只有以下情况才询问我：

1. 两种以上科学上都合理、但会明显改变项目含义的选择；
2. 缺少任何文献都无法合理决定的关键生物参数；
3. 当前 64 GB RAM / RTX 5060 8 GB 等硬件形成已经测量确认的硬瓶颈，而且不存在保持模型语义的工程解决方案；
4. 需要改变上述最终目标。

不要把“跑一个小 demo”当完成。
不要因为计算量大自动删 neuron/synapse。
不要改其他项目或其他 Conda 环境。

允许为了 debug 用少量数据，但最终验收必须回到全部 MaleCNS。

==================================================
十、最终成果
==================================================

最终我应该能够启动一个程序，看到：

左侧：
真实运行的 MuJoCo 数字果蝇和环境。

右侧：
完整 MaleCNS 的 3D morphology 与实时活动。

下方/其他面板：
左右复眼输入、感觉状态、motor outputs、身体 telemetry。

果蝇能够在环境里：

- 站立
- 行走
- 转向
- 与地面和障碍碰撞
- 感知光
- 感知气味
- 感知腿和身体状态
- 对环境产生由 CNS 自己决定的反应
- 在实现有效飞行模型后起飞和飞行

用户应能暂停 simulation，点击/搜索 neuron，查看其真实 morphology、synapses、连接、递质和动态状态。

实验环境可切换到：
- phototaxis
- Y-maze odor choice
- obstacle avoidance
- looming stimulus
- general sandbox

最重要的一条：

不要预设果蝇应该做什么。

给完整 MaleCNS 一个尽可能有根据的身体和感觉系统，让行为从真实 connectome + 当前有依据的神经动力学 + 物理环境闭环中自然产生。

从现在开始直接实施。
