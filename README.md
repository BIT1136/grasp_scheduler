# 抓取调度器

部分代码来自vgn

## 安装依赖

    conda create -n grasp_scheduler python=3.10 

可以分别安装另外三个软件包中的依赖，或使用以下命令安装支持所有四个软件包的所有依赖：

    

## 运行节点

    roslaunch grasp_scheduler node.launch

## 状态机可视化

    rosrun smach_viewer smach_viewer.py
    