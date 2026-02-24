# Unit2Wav thưc hiện chuyển đổi dữ liệu từ chuoxi unit sang wav 
## ơ đây  thực hiện cả 2 nguồn đó là target và source trong  đồ án dùng cặp ngôn ngữ ANh_việt 
## Muc tiêu là training lại vocodẻr cho các cặp dữ liệu chưa có sẵn (việt) , dùng framework https://github.com/jik876/hifi-gan  , để training lại vocoder 

## Train
- input train  : file định dạng chuẩn theo yêu cầu của fihi_gan  
[file_input](image/vocoder_image.png)
- output : gồm 2 file sau khi train là : g_xxxxxx ,  d_xxxxxx  (g_xxxx là file để chuyển đổi dữ liệu unit -> wav ,  dile d_xxxxx là file có thể dùng để fine_tune tiếp)
### thực hiện triển khai unit2wav từng nguồn 
-  lệnh chuẩn bị dữ liệu cho train : 
    + source : python src/Unit2Wav/processing.py --lang source
    + target : python src/Unit2Wav/processing.py --lang target

-  confing : trong phần config  chỉnh tham số   trong file [config](src/Unit2Wav/config.json)
    + source : (chỉnh dòng 41 , 42 trong file) 

    "input_training_file": "/mnt/e/AI/khanh/src/Unit2Wav/processed_data/source/train.manifest",
    "input_validation_file": "/mnt/e/AI/khanh/src/Unit2Wav/processed_data/source/valid.manifest"


    + target : (chỉnh dòng 41 , 42 trong file )
    
    "input_training_file": "/mnt/e/AI/khanh/src/Unit2Wav/processed_data/target/train.manifest",
    "input_validation_file": "/mnt/e/AI/khanh/src/Unit2Wav/processed_data/target/valid.manifest"


-  lênh train : 
    + source : python src/Unit2Wav/train.py --lang source
    + target : python src/Unit2Wav/train.py --lang target



## Infer 
-  input : copy chuỗi unit vào đường dẫn 
    + source : [link gắn unit](/mnt/e/AI/khanh/final/unit2wav/target/predicted_unit.txt)
    + target : [link gắn unit](/mnt/e/AI/khanh/final/unit2wav/source/predicted_unit.txt)

-  output : 
    + source : [wav](/mnt/e/AI/khanh/final/unit2wav/target/predicted_wav)
    + target : [wav](/mnt/e/AI/khanh/final/unit2wav/target/predicted_wav)

- lệnh chạy  ; 
    + source : python src/Unit2Wav/infer.py --lang source 
    + target : python src/Unit2Wav/infer.py --lang target 

