from ultralytics import YOLO

modelo = YOLO('yolov8n.pt')

#modelo.predict(source='0', show=True)


# Realiza a detecção de objetos na webcam
resultado = modelo.predict(source='0')

# Obtém as previsões
previsoes = resultado.pandas().xyxy[0]

# Verifica se a classe "bicicleta" está presente nas previsões
bicicleta_presente = any(previsoes['name'] == 'bicycle')

if bicicleta_presente:
    print('Bicicleta detectada!')
else:
    print('Nenhuma bicicleta detectada.')

