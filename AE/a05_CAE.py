# 2번 카피해서 복붙
# cnn으로 딥하게 구성
# 2개의 모델을 구성하는데 하나느 ㄴ기본적 오토인코더
# 다른하나는 딥하게 만든구성
# 2개의성능비교

con2d
maxpool
con2d
maxpool
con2d -> encoder

conv2d
upsamp 2d
con2d
up 2d
con2d
up2
con2d(1,) ->decoder