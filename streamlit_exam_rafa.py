import random
import math
import streamlit as st
import matplotlib.pyplot as plt

class MonteCarloIntegrator:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.reset()
    
    #Método de reiniciar variables
    def reset(self):
        self.x_values = []
        self.f_values = []
        self.areas = []
        self.integral_estimate = 0
    
    #Función a estimar
    def f(self, x):
        return 1.0 / (math.exp(x) + math.exp(-x))
    
    #Método que hace el Montecarlo completo
    def integrate(self, a, b, n):
        self.reset()
        
        #Creación de muestras
        sample = []
        for _ in range(n):
            x = random.uniform(a, b)
            sample.append(x)
        self.x_values = sample.copy()
        
        #Calculación de valores de muestras
        f_values = []
        for x in sample:
            fx = self.f(x)
            f_values.append(fx)
        self.f_values = f_values.copy()
        
        width = (b - a) / n
        total = 0
        areas = []
        
        #Sumatoria de métodos
        for fx in f_values:
            area = width * fx
            areas.append(area)
            total += area
        
        self.areas = areas
        self.integral_estimate = total
        return total
        
    def get_results(self):
        return {
            'x_values': self.x_values,
            'f_values': self.f_values,
            'areas': self.areas,
            'estimate': self.integral_estimate,
            'sample_size': len(self.x_values)
        }

# Interfaz Streamlit
st.title("Integración Monte Carlo")
st.latex(r"\frac{1.0}{e^x + e^{-x}}")
# Controles de entrada
col1, col2, col3 = st.columns(3)
with col1:
    a = st.number_input("Límite inferior (a)", value=0.0, step=0.1)
with col2:
    b = st.number_input("Límite superior (b)", value=1.0, step=0.1)
with col3:
    n = st.number_input("Número de muestras (n)", value=10000, min_value=10, step=10)

seed = st.number_input("Semilla aleatoria (opcional)", value=None, placeholder="Deja vacío para aleatorio")

# Botón para calcular
if st.button("Calcular Integral"):
    # Crear instancia y calcular
    integrator = MonteCarloIntegrator(seed if seed is not None else None)
    integral = integrator.integrate(a, b, int(n))
    results = integrator.get_results()
    
    # Mostrar resultado
    st.subheader(f"Estimación de la integral: {integral:.6f}")
    
    # Crear gráficos
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gráfica de puntos aleatorios
    axes[0, 0].scatter(results['x_values'], results['f_values'], alpha=0.5, s=10)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('f(x)')
    axes[0, 0].set_title('Valores Aleatorios Generados')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfica de barras para alturas y áreas
    t = 100 if int(n/100) >= 1 else n

    indices = list(range(min(int(t), len(results['x_values']))))
    if indices:
        x_pos = [i for i in indices]
        axes[0, 1].bar(x_pos, [results['f_values'][i] for i in indices], 
                      alpha=0.6, label='Alturas (f(x))', color='blue')
        axes[0, 1].bar(x_pos, [results['areas'][i] for i in indices], 
                      alpha=0.4, label='Áreas', color='red')
        axes[0, 1].set_xlabel('Índice de muestra')
        axes[0, 1].set_ylabel('Valor')
        axes[0, 1].set_title(f'Primeras {int(t)} Alturas y Áreas')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfica de sumatoria acumulativa de áreas
    cumulative_areas = []
    current_sum = 0
    for area in results['areas']:
        current_sum += area
        cumulative_areas.append(current_sum)
    
    axes[1, 0].plot(range(len(cumulative_areas)), cumulative_areas, 'g-', linewidth=2)
    axes[1, 0].axhline(y=integral, color='r', linestyle='--', alpha=0.5, label=f'Valor final: {integral:.6f}')
    axes[1, 0].set_xlabel('Número de iteración')
    axes[1, 0].set_ylabel('Suma acumulada de áreas')
    axes[1, 0].set_title('Convergencia de la Integral')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histograma de áreas
    axes[1, 1].hist(results['areas'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=integral/n, color='r', linestyle='--', 
                       label=f'Área promedio: {integral/n:.6f}')
    axes[1, 1].set_xlabel('Área de cada muestra')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title('Distribución de Áreas')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Mostrar datos numéricos
    with st.expander("Ver datos numéricos"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Primeros 10 valores de x:**")
            st.write(results['x_values'][:10])
        with col2:
            st.write("**Primeras 10 áreas:**")
            st.write(results['areas'][:10])
        
        st.write(f"**Estadísticas:**")
        st.write(f"- Muestra total: {results['sample_size']}")
        st.write(f"- Valor máximo de f(x): {max(results['f_values']):.6f}")
        st.write(f"- Valor mínimo de f(x): {min(results['f_values']):.6f}")
        st.write(f"- Área máxima: {max(results['areas']):.6f}")
        st.write(f"- Área mínima: {min(results['areas']):.6f}")