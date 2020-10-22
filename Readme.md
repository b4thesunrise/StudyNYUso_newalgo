#### Database：

- 使用的文件列表（da和rt文件并不在数量上对齐）: 以filelist.pkl 结尾的文件
- 整合后的数据，分别有原始版，调整版，加了时间特征版: 以*df.csv结尾的文件
- npy文件：提前计算好的策略风险矩阵，用于提升运行时效率

#### Environment：基于数据搭建的新环境：只考虑出的数量

#### Experiment_Log：sac和ddpg的实验结果

#### IPython_Notebook

- Crawler.ipynb: 爬取数据
- Processing.ipynb: 做数据的格式处理
- Analysis.ipynb: 做数据的周期性分析，相关性分析，可用性分析

#### .PY：sac与ddpg的基础实验