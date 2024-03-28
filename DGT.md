
## DGT
### Calulate the beam label of each dataset
<details><summary>Code</summary>

For synLiDAR:
```
python dgt_utils/get_beam_label.py \
        --data-path ~/dataset/SynLiDAR/sub_dataset \
        --sequences 00 01 02 03 04 05 06 07 08 09 10 11 12 \
        --data-name SynLiDAR \
        --num-beams 64
```


#### Note: we do not need calculate the SemanticKITTI and SemanticPOSS beam label. However, you can obtain them by the following code.

For SemanticKITTI:
```
python dgt_utils/get_beam_label.py \
        --data-path ~/dataset/semanticKITTI/dataset/sequences \
        --sequences 00 01 02 03 04 05 06 07 09 10 \
        --data-name SemanticKITTI \
        --num-beams 64
```

For SemanticPOSS:
```
python dgt_utils/get_beam_label.py \
        --data-path ~/dataset/semanticPOSS/dataset/sequences \
        --sequences 00 01 02 04 05 \
        --data-name SemanticPOSS \
        --num-beams 40
```

</details>

###  Calulate the downsampled density of each dataset
<details><summary>Code</summary>

For SynLiDAR to SemanticPOSS:

```
python dgt_utils/hist_DownSampled_syn_dataloadMode.py \
        --data-path ~/dataset/SynLiDAR/sub_dataset \
        --sequences 00 01 02 03 04 05 06 07 08 09 10 11 12 \
        --data-name SynLiDAR \
        --tgt-data-name SemanticPOSS
```
Results:
[23354.038508064517, 10496.810433467743, 3684.935433467742, 1767.5733870967742, 911.376814516129, 575.5527217741935, 364.83311491935484, 238.79269153225806, 171.72595766129032, 128.75640120967742]

For SynLiDAR to nuScenes:
```
python dgt_utils/hist_DownSampled_syn_dataloadMode.py \
        --data-path ~/dataset/SynLiDAR/sub_dataset \
        --sequences 00 01 02 03 04 05 06 07 08 09 10 11 12 \
        --data-name SynLiDAR \
        --tgt-data-name nuScenes
```
Results:
[17786.377318548388, 10493.941431451613, 3684.935433467742, 1767.5733870967742, 911.376814516129, 575.5527217741935, 364.83311491935484, 238.79269153225806, 171.72595766129032, 128.75640120967742]
</details>

### Calulate the density of each dataset
```python dgt_utils/hist_syn.py ```

#### We also implement a more efficient version, which is faster than the above, and we recommend using this.

```python dgt_utils/hist_syn_dataloadMode.py ```
<details><summary>Code</summary>

For SynLiDAR:
```
python dgt_utils/hist_syn_dataloadMode.py \
        --data-path ~/dataset/SynLiDAR/sub_dataset \
        --sequences 00 01 02 03 04 05 06 07 08 09 10 11 12 \
        --data-name SynLiDAR
```

Results:
[34404.903377016126, 21186.506401209677, 7464.479989919355, 3618.0602318548385, 1907.870816532258, 1166.4787802419355, 758.060685483871, 520.3017641129032, 365.8337197580645, 266.6142137096774]

For SemanticKITTI:
```
python dgt_utils/hist_syn_dataloadMode.py \
        --data-path ~/dataset/semanticKITTI/dataset/sequences \
        --sequences 00 01 02 03 04 05 06 07 09 10 \
        --data-name SemanticKITTI
```
Results:
[43156.416884474645, 28572.189440669106, 8547.528280188186, 3359.0474124411917, 1657.9704652378464, 888.4075274438055, 535.0811291165709, 327.98860428646105, 3.2445373758494513, 0.0]


For SemanticPOSS:
```
python dgt_utils/hist_syn_dataloadMode.py \
        --data-path ~/dataset/semanticPOSS/dataset/sequences \
        --sequences 00 01 02 04 05 \
        --data-name SemanticPOSS
```
Results: 
[11928.82274919614, 22433.851688102895, 14100.801848874598, 6775.664389067524, 3469.734726688103, 1925.7777331189711, 1209.1840836012861, 762.4602090032155, 567.3782154340836, 377.77371382636653]

For nuScenes:
```
python dgt_utils/hist_syn_dataloadMode.py \
        --data-path ~/dataset/Sk2Nusc_KittiFormat_10 \
        --sequences 01 \
        --data-name nuScenes
```
Results:
[11656.442836829008, 5282.730465694987, 2152.471667259154, 1111.5349448986847, 621.7337362246711, 351.21048702452896, 191.19701386420192, 95.24884464984002, 46.488801990757196, 23.40298613579808]
</details>
