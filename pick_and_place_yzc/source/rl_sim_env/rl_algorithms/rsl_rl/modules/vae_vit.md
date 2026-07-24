%% https://www.mermaidchart.com/ %%
flowchart TD
  %% ===== 输入 =====
  A1["Proprio-history(N,P,45)"]
  A2["PointCloud-history(N,Q,M,K,3)"]

  %% ===== Proprioceptive 分支 =====
  A1 --> B1["MLP&nbsp;(45→d)"]
  B1 --> B2["Time PEEmbedding"]
  B2 --> C1["z_prop(N,P,d)"]

  %% ===== Exteroceptive 分支 =====
  A2 --> D0["Reshape&nbsp;&amp;&nbsp;Permute(NQM,3,K)"]
  D0 --> D1["Relative coord.pts − centers_local"]
  D1 --> D2["Shared MLPConv1d"]
  D2 --> D3["Confidence Filterstd → conf ×"]
  D3 --> D4["MaxPool K→1"]
  D4 --> D5["Reshape → (N,Q,M,d)"]
  D5 --> D6["Spatial PEcenter_mlp"]
  D6 --> D7["Time PEidx_pc"]
  D7 --> C2["Flatten(N,Q·M,d)"]

  %% ===== Token 拼接 =====
  C1 & C2 --> E1["Concat tokens"]
  E1 --> E2["Insert CLS"]

  %% ===== Transformer =====
  E2 --> F1["Transformer Encoder"]
  F1 --> F2["Norm + Dropout"]
  F2 --> G1["CLS token"]
  F2 --> G2["MaxPoolrest tokens"]
  G1 & G2 --> H1["Concat (N,2d)"]

  %% ===== GRU (stateful) =====
  H1 --> I1["GRU"]
  I1 --> I2["Norm + Dropout"]

  %% ===== Multi-head VAE =====
  I2 --> J1["μ / logσ²vel • mass • g • h"]
  J1 --> J2["Reparameterise"]
  J2 --> K1["Concat latent(N,1,84)"]

  %% ===== 解码 =====
  K1 --> L1["Obs Decoder → 45"]
  K1 --> L2["Heightmap Decoder → 2001"]

