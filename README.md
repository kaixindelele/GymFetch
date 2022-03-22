# GymFetch-插孔开抽屉任务介绍

## 前言：
最近做HER相关对比实验，需要几个验证仿真环境，所以仿照原版gym-fetch的封装格式，借用了metaworld的素材，为了和push，pick有所区别，所以重新搭建了两个环境，一个是随机插孔任务，一个是开抽屉任务。前者目前看来是符合需要的。后者面临着穿模的问题，后面会有所讨论。

开源链接：[https://github.com/kaixindelele/GymFetch](https://github.com/kaixindelele/GymFetch)


## 插孔任务介绍：
总所周知，`mujoco`是不能仿真凹形体的，所以如果导入一个3D建模好的插孔盒子，是无法正常模拟的，插入孔中都会出现碰撞的问题。

而插孔任务其实很多论文中都提到了，比如朱玉可的19年那篇best paper, 比如GUAPO算法：

![在这里插入图片描述](https://img-blog.csdnimg.cn/f8de6587c2144fe48043601d6b002417.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaGVoZWRhZGFx,size_10,color_FFFFFF,t_70,g_se,x_16)


![在这里插入图片描述](https://img-blog.csdnimg.cn/be685439c3d14a56939f2779dca6c806.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaGVoZWRhZGFx,size_16,color_FFFFFF,t_70,g_se,x_16)


但好用的，开源的插孔仿真环境，我还真的没怎么找到过，就连robosuite里面都没搭建一个好的插孔任务，我是没想到的，待会儿发个邮件问问他们。目前能找到的都是四个长方形拼成的一个带孔盒子：
![在这里插入图片描述](https://img-blog.csdnimg.cn/44ec3a39f8c846bfbe21e9aaaa445a36.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaGVoZWRhZGFx,size_20,color_FFFFFF,t_70,g_se,x_16)


现在有另一个问题，这里的四个box都是没有自由度的，mujoco的box 几何体如果加了joint自由度的话，那么被夹爪接触了之后，会发生偏移，哪怕是将关节阻尼`damping='10000000000'`都会从缝隙中插进去。

所以这四个box必须得是无自由度的。

这样的话，带孔盒子和孔就是固定的，不利于任务难度的提高。

我必须要随机初始化孔的中心位置，然后根据孔的中心位置，安排box的位置，这样的话，就需要一个坐标生成函数，一个每次reset时，设置box位置的函数。

box坐标生成函数：输入孔的中心坐标xy，和孔的大小，整个盒子的长度一半。
输出四个box的坐标和大小。

```python
def reset_hole(self, hole_center, half_hole_size=0.02, hsw1y=0.1):
    cx, cy = hole_center
    hsw1x = (hsw1y - half_hole_size) / 2.0
    w1x = cx - hsw1x - half_hole_size
    w1y = cy

    hsw2x = (2*hsw1y-2*hsw1x)/2.0
    hsw2y = (hsw1y-half_hole_size)/2.0
    w2x = w1x + hsw1x + (2*hsw1y-2*hsw1x)/2.0
    w2y = w1y - hsw1y + (hsw1y-half_hole_size)/2.0

    hsw3x = hsw2x
    hsw3y = hsw2y
    w3x = w2x
    w3y = w1y + hsw1y - (hsw1y-half_hole_size)/2.0

    hsw4x = (hsw1y-half_hole_size)/2
    hsw4y = half_hole_size
    w4x = cx + half_hole_size + (hsw1y - half_hole_size) / 2.0
    w4y = cy
    return (w1x, w1y), (w2x, w2y), (w3x, w3y), (w4x, w4y)
```


设定body的pos，mujoco中除了set_joint_pos函数外，还能直接给body_pos设定值：

```python
self.sim.model.body_pos[self.sim.model.body_name2id('w4')] = np.array(
            [w4x, w4y, self.sim.model.body_pos[self.sim.model.body_name2id('w4')][2]])
```
这样就可以随机初始化孔的位置了。

## 插孔任务observation的设置：
关于环境观察值的输出格式，分为
'observation': obs.copy(), 除了desired_goal之外的都在里面

'grip_goal': grip_pos.copy(), 夹爪的末端坐标xyz

'achieved_goal': achieved_goal.copy(), desired_goal if grip2desire > 2cm else grip_goal，即离得远时，是目标点的坐标（加上移偏置），离的近了，直接变为夹爪坐标。即假设有一个虚拟物块在目标上方，到了之后，贴着夹爪随着夹爪移动。

'desired_goal': self.goal.copy(),

上面基本上说清楚了观察值。

关于奖励函数的设置，仍然是稀疏奖励，但是将距离阈值变成了1cm（如果难度过高，回头再调）。

## 最终效果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/6f6229796b264ebab4b3a441c35072c8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaGVoZWRhZGFx,size_20,color_FFFFFF,t_70,g_se,x_16)

## 抽屉任务：
### 横版：
![在这里插入图片描述](https://img-blog.csdnimg.cn/53887119af214873a1d6049f24d4e69e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaGVoZWRhZGFx,size_20,color_FFFFFF,t_70,g_se,x_16)
### 竖版：
![在这里插入图片描述](https://img-blog.csdnimg.cn/999f3ed00e6c47a09e4799201656ca07.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaGVoZWRhZGFx,size_20,color_FFFFFF,t_70,g_se,x_16)
我想的是将整个抽屉设置成刚体，夹爪穿不过去的那种，但是现在出现各种奇怪的问题，如果将里面的抽屉joint damp设置的低，那么就会自己抖动，如果设置damp大了，那么夹爪会想着直接穿过外壳，直接拉着里面的抽屉往外动。

关节limited属性也不起作用，有点迷惑。回头再调调。

## 联系方式：
ps: 欢迎做强化的同学加群一起学习：

深度强化学习-DRL：799378128

Mujoco建模：818977608

欢迎玩其他物理引擎的同学一起玩耍~

欢迎关注知乎帐号：[未入门的炼丹学徒](https://www.zhihu.com/people/heda-he-28)

CSDN帐号：[https://blog.csdn.net/hehedadaq](https://blog.csdn.net/hehedadaq)

我的个人博客：

[未入门的炼丹学徒](http://metair.top/)

网址非常好记：`metair.top`

极简spinup+HER+PER代码实现：[https://github.com/kaixindelele/DRLib](https://github.com/kaixindelele/DRLib)

