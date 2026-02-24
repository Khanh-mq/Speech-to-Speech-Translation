# Module Wav2Unit: Audio Preprocessing Pipeline **
### Module này chịu trách nhiệm biến đổi dữ liệu âm thanh thô (.wav) thành chuỗi các đơn vị định danh (Unit IDs) phục vụ cho huấn luyện mô hình Speech-to-Unit Translation (S2UT).

## Chức năng chính
1. **Tạo Manifest**: Tự động quét và lập chỉ mục dữ liệu âm thanh dưới dạng file .tsv.
    -  input : file âm thanh có đuôi .wav (và đã đưa về dạng chuẩn 16k)
    [_0000001.wav](/mnt/e/AI/khanh/image/image_1.png)
    - output : xuất ra 1 file có định dạng là .tsv 
    [file.tsv](/mnt/e/AI/khanh/image/file_tsv.png)
2. **Trích xuất đặc trưng (Feature Extraction)**: Sử dụng mô hình mHubert để biến âm thanh thành các vector đặc trưng số học.
    - input: đầu vào là file tsv được quyets và  lập chỉ mục ở phần tạo manifest 
    - output : định dạng file gồm file có đuôi .npy và .len
    [hình ảnh 2 file](/mnt/e/AI/khanh/image/file_extraction.png)

3. **Huấn luyện K-means**: Tự học bộ tâm cụm  cho các ngôn ngữ chưa có sẵn model (ví dụ: Tiếng Việt).
    -  input: file .npy vừa trích xuất đặc chưng ở phần trên 
    - output: xuất ra 1 file có định dạng .bin 
    [file .bin](/mnt/e/AI/khanh/image/file_kmean.png) 

4. **Quantization**: Mã hóa các đặc trưng vector thành chuỗi số nguyên (Unit IDs).
    - input : file model huấn luyện kmean ở trên hoăc là file có sẵn 
    - output : file được chuyển hóa thành unit 
    [unit](/mnt/e/AI/khanh/image/unit.png)


## thực hiện chạy code 
### lần đầu tiên chạy code sinh ra uint của tiếng anh và tiếng việt đồng thời tạo kmean cho ngôn ngữ đích(tiếng việt) hiện chưa có kmean 
- lệnh chạy trên data source : python src/Wav2Unit/train.py --source  --all (gắn cờ )
- lệnh chạy trên data target : python src/Wav2Unit/train.py --target  --all (gắn cờ )

## infer 
- chuẩn bị dữ liệu : copy 1 file wav vào bên trong final/wav2unit/source(target)/input đặt tên file (input.wav)
- thực hiện chạy test dữ liệu 
    + source : python  src/Wav2Unit/infer.py --lang source
    + target : python  src/Wav2Unit/infer.py --lang target











