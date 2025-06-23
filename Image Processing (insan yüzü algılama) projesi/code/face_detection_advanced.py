import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        # Kamera parametreleri
        self.KNOWN_DISTANCE = 50.0  # cm
        self.KNOWN_FACE_WIDTH = 15.0  # cm
        self.focal_length = None
        
        # Ten rengi için çoklu renk uzayı değerleri
        # HSV renk uzayı için
        self.hsv_ranges = {
            'lower1': np.array([0, 30, 60], dtype=np.uint8),
            'upper1': np.array([20, 150, 255], dtype=np.uint8),
            'lower2': np.array([170, 30, 60], dtype=np.uint8),  # Kırmızı spektrumun diğer ucu
            'upper2': np.array([180, 150, 255], dtype=np.uint8)
        }
        
        # YCrCb renk uzayı için
        self.ycrcb_ranges = {
            'lower': np.array([0, 135, 85], dtype=np.uint8),
            'upper': np.array([255, 180, 135], dtype=np.uint8)
        }
        
        # Mesafe yumuşatma
        self.distance_history = []
        self.HISTORY_SIZE = 5
        
        # Morfolojik işlemler için kerneller
        self.kernel_small = np.ones((3,3), np.uint8)
        self.kernel_large = np.ones((5,5), np.uint8)

    def create_skin_mask(self, frame):
        """Gelişmiş ten rengi tespiti"""
        # Görüntüyü yumuşat
        blurred = cv2.GaussianBlur(frame, (5,5), 0)
        
        # Kontrast iyileştirme
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # HSV renk uzayında ten rengi tespiti
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.hsv_ranges['lower1'], self.hsv_ranges['upper1'])
        mask2 = cv2.inRange(hsv, self.hsv_ranges['lower2'], self.hsv_ranges['upper2'])
        hsv_mask = cv2.bitwise_or(mask1, mask2)
        
        # YCrCb renk uzayında ten rengi tespiti
        ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YCrCb)
        ycrcb_mask = cv2.inRange(ycrcb, self.ycrcb_ranges['lower'], self.ycrcb_ranges['upper'])
        
        # İki maskeyi birleştir
        combined_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
        
        # Gürültü temizleme
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.kernel_small)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, self.kernel_large)
        
        # Son yumuşatma
        final_mask = cv2.GaussianBlur(cleaned_mask, (5,5), 0)
        _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
        
        return final_mask

    def detect_face(self, frame):
        """Yüz tespiti ve analizi"""
        try:
            # Ten rengi maskesi oluştur
            skin_mask = self.create_skin_mask(frame)
            
            # Maskeyi BGR'ye dönüştür (görselleştirme için)
            mask_bgr = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
            
            # Konturları bul
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, None, mask_bgr
            
            # En büyük konturu al
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            
            # Alan kontrolü
            if area < 10000:  # Minimum yüz alanı
                return None, None, mask_bgr
            
            # Konveks örtü analizi
            hull = cv2.convexHull(max_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area
            
            # Şekil analizi
            peri = cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)
            
            # Yüz şekli kontrolü
            if len(approx) < 8 or len(approx) > 20:  # Yüz şekli için nokta sayısı
                return None, None, mask_bgr
                
            if solidity < 0.8:  # Yüz doluluk oranı
                return None, None, mask_bgr
            
            # Sınırlayıcı dikdörtgen
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # En-boy oranı kontrolü
            aspect_ratio = float(w) / h
            if aspect_ratio > 1.3 or aspect_ratio < 0.7:  # Yüz oranı
                return None, None, mask_bgr
            
            # Tespit edilen yüzü maskede göster
            cv2.drawContours(mask_bgr, [max_contour], -1, (0, 255, 0), 2)
            cv2.rectangle(mask_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(mask_bgr, (x + w//2, y + h//2), 4, (0, 0, 255), -1)
            
            return (x, y, w, h), max_contour, mask_bgr
            
        except Exception as e:
            print(f"Yüz tespit hatası: {str(e)}")
            return None, None, None

    def calculate_distance(self, face_width):
        """Mesafe hesaplama"""
        try:
            if self.focal_length is None:
                self.focal_length = (face_width * self.KNOWN_DISTANCE) / self.KNOWN_FACE_WIDTH
                print(f"Kalibrasyon tamamlandı. Focal length: {self.focal_length:.2f}")
                return self.KNOWN_DISTANCE
            
            distance = (self.KNOWN_FACE_WIDTH * self.focal_length) / face_width
            
            # Mesafe yumuşatma
            self.distance_history.append(distance)
            if len(self.distance_history) > self.HISTORY_SIZE:
                self.distance_history.pop(0)
            
            # Aykırı değerleri filtrele
            if len(self.distance_history) >= 3:
                sorted_distances = sorted(self.distance_history)
                filtered_distances = sorted_distances[1:-1]  # En düşük ve en yüksek değerleri at
                return np.mean(filtered_distances)
            
            return np.median(self.distance_history)
            
        except Exception as e:
            print(f"Mesafe hesaplama hatası: {str(e)}")
            return None

    def process_frame(self, frame):
        """Ana işleme fonksiyonu"""
        try:
            result = frame.copy()
            
            # Yüz tespiti
            face_data = self.detect_face(frame)
            
            if face_data is None or face_data[0] is None:
                cv2.putText(result, "Yuz tespit edilemedi", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return result, np.zeros_like(frame)
            
            rect, contour, mask = face_data
            x, y, w, h = rect
            
            # Mesafe hesapla
            distance = self.calculate_distance(w)
            
            if distance is not None:
                # Renk seçimi (mesafeye göre)
                if distance < 30:
                    color = (0, 0, 255)  # Kırmızı (çok yakın)
                    status = "COK YAKIN!"
                elif distance > 100:
                    color = (0, 255, 255)  # Sarı (çok uzak)
                    status = "COK UZAK!"
                else:
                    color = (0, 255, 0)  # Yeşil (uygun mesafe)
                    status = "UYGUN MESAFE"
                
                # Yüz çerçevesi
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
                cv2.drawContours(result, [contour], -1, (255, 0, 0), 2)
                
                # Merkez nokta
                center_x = x + w//2
                center_y = y + h//2
                cv2.circle(result, (center_x, center_y), 4, (0, 0, 255), -1)
                
                # Bilgi paneli
                cv2.putText(result, f"Mesafe: {distance:.1f} cm", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(result, status, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(result, f"Genislik: {w}px", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return result, mask
            
        except Exception as e:
            print(f"Frame işleme hatası: {str(e)}")
            return frame, np.zeros_like(frame)

def main():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Kamera açılamadı!")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        detector = FaceDetector()
        print("Gelişmiş yüz tespit sistemi başlatıldı")
        print("Kalibrasyon için yüzünüzü 50cm uzaklıkta tutun")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            result, mask = detector.process_frame(frame)
            
            cv2.imshow('Yuz Tespiti', result)
            cv2.imshow('Ten Maskesi', mask)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
    
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 