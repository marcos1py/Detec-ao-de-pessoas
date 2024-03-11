import cv2

# Carregar o classificador em cascata para detectar corpos inteiros
detector_corpo_inteiro = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Carregar a imagem
imagem = cv2.imread("pessoas_corpos_inteiros.jpg")

# Converter a imagem para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Detectar corpos inteiros na imagem
deteccoes = detector_corpo_inteiro.detectMultiScale(imagem_cinza)

# Desenhar retângulos ao redor das detecções
for (x, y, w, h) in deteccoes:
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0,0,255), 2)

# Mostrar a imagem com as detecções
cv2.imshow("Detecção de Corpos Inteiros", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
