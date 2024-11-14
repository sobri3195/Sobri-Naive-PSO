import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import random

class ParticleSwarmOptimization:
    def __init__(self, n_particles, n_features):
        self.n_particles = n_particles
        self.n_features = n_features
        
        # Inisialisasi parameter PSO
        self.w = 0.7  # Inertia weight
        self.c1 = 2.0  # Cognitive weight
        self.c2 = 2.0  # Social weight
        
        # Inisialisasi posisi dan kecepatan partikel
        self.positions = np.random.rand(n_particles, n_features)
        self.velocities = np.zeros((n_particles, n_features))
        
        # Inisialisasi personal best dan global best
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.zeros(n_particles)
        self.gbest_position = None
        self.gbest_score = float('-inf')

    def update_particle(self, particle_idx):
        r1, r2 = random.random(), random.random()
        
        # Update kecepatan
        self.velocities[particle_idx] = (self.w * self.velocities[particle_idx] +
                                       self.c1 * r1 * (self.pbest_positions[particle_idx] - self.positions[particle_idx]) +
                                       self.c2 * r2 * (self.gbest_position - self.positions[particle_idx]))
        
        # Update posisi
        self.positions[particle_idx] += self.velocities[particle_idx]
        
        # Batasi posisi antara 0 dan 1
        self.positions[particle_idx] = np.clip(self.positions[particle_idx], 0, 1)

    def optimize(self, X, y, n_iterations):
        for i in range(n_iterations):
            for j in range(self.n_particles):
                # Pilih fitur berdasarkan posisi partikel
                selected_features = self.positions[j] > 0.5
                if not any(selected_features):  # Jika tidak ada fitur yang terpilih
                    selected_features[0] = True  # Pilih minimal satu fitur
                
                # Evaluasi performa menggunakan Naive Bayes
                X_selected = X[:, selected_features]
                nb = GaussianNB()
                scores = cross_val_score(nb, X_selected, y, cv=5)
                current_score = np.mean(scores)
                
                # Update personal best
                if current_score > self.pbest_scores[j]:
                    self.pbest_scores[j] = current_score
                    self.pbest_positions[j] = self.positions[j].copy()
                
                # Update global best
                if current_score > self.gbest_score:
                    self.gbest_score = current_score
                    self.gbest_position = self.positions[j].copy()
                
                # Update partikel
                self.update_particle(j)
        
        return self.gbest_position, self.gbest_score

class OptimizedNaiveBayes:
    def __init__(self, n_particles=30, n_iterations=100):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.selected_features = None
        self.nb_classifier = GaussianNB()
        
    def fit(self, X, y):
        # Inisialisasi PSO
        pso = ParticleSwarmOptimization(self.n_particles, X.shape[1])
        
        # Optimasi feature selection menggunakan PSO
        self.selected_features, best_score = pso.optimize(X, y, self.n_iterations)
        
        # Training Naive Bayes dengan fitur terpilih
        X_selected = X[:, self.selected_features > 0.5]
        self.nb_classifier.fit(X_selected, y)
        
        return self
    
    def predict(self, X):
        # Prediksi menggunakan fitur terpilih
        X_selected = X[:, self.selected_features > 0.5]
        return self.nb_classifier.predict(X_selected)
    
    def predict_proba(self, X):
        X_selected = X[:, self.selected_features > 0.5]
        return self.nb_classifier.predict_proba(X_selected)

# Fungsi untuk memuat dan memproses data
def load_and_preprocess_data(file_path):
    # Baca dataset
    data = pd.read_csv(file_path)
    
    # Pisahkan fitur dan target
    X = data.drop('target', axis=1)  # Sesuaikan dengan nama kolom target Anda
    y = data['target']
    
    # Standardisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Fungsi utama
def main():
    # Load data
    X, y = load_and_preprocess_data('path_to_your_dataset.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inisialisasi dan training model
    model = OptimizedNaiveBayes(n_particles=30, n_iterations=100)
    model.fit(X_train, y_train)
    
    # Evaluasi model
    y_pred = model.predict(X_test)
    
    # Tampilkan hasil
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Tampilkan fitur yang terpilih
    selected_features = np.where(model.selected_features > 0.5)[0]
    print("\nSelected Features:", selected_features)

if __name__ == "__main__":
    main()
