import cv2
import numpy as np

imagen = cv2.imread('C:/Users/LaboratorioLTI5/Downloads/circulos.jpg')
#cv2.imshow('imagen', imagen)
gris =cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)

circulos = cv2.HoughCircles(
    gris, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
    param1=50, param2=30, minRadius=10, maxRadius=50
)
if circulos is not None:
    circulos = np.round(circulos[0 , :]).astype("int")
    escala_referencial = 20.0
    referencial_circulo = circulos[0]

    for (x, y, r) in circulos:
        cv2.circle(imagen, (x, y), r, (0, 255, 0), 2)
        cv2.circle(imagen, (x, y), 2, (0, 0, 255), 3)
        pixel_distancia = np.sqrt((x - referencial_circulo[0]) ** 2 + (y -referencial_circulo[1]) ** 2)
        escala = escala_referencial / referencial_circulo[2]
        real_distancia = pixel_distancia * escala

        cv2.line(imagen, (referencial_circulo[0], referencial_circulo[1]), (x,y), (255,0,0))
        cv2.putText(imagen, f"{real_distancia: .2f} mm", (x, y -10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2)

    cv2.circle(imagen, (referencial_circulo[0], referencial_circulo[1]), referencial_circulo[2], [255,0,0], 2)
    cv2.putText(imagen, "Refencia", (referencial_circulo[0] - 40, referencial_circulo[1]- referencial_circulo[2] -10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
cv2.imshow('Distacia entre circulos', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()