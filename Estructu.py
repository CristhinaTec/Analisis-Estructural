import numpy as np
import math
import plotly.graph_objects as go

# ==============================================================================
# MÓDULO DE CÁLCULO DE MATRICES DE RIGIDEZ
# ==============================================================================

def obtener_matriz_rigidez_armadura(E, A, L, c, s):
    """
    Calcula la matriz de rigidez global 4x4 para un elemento de armadura.
    ✅ CORREGIDO: Los signos de los términos 'cs' han sido ajustados
    para coincidir con la formulación teórica T.T @ K_local @ T.
    """
    k = (A * E) / L
    c2 = c * c
    s2 = s * s
    cs = c * s
    
    # Esta es la matriz correcta resultante de la transformación

    return k * np.array([
        [c2, cs, -c2, -cs], 
        [cs, s2, -cs, -s2], 
        [-c2, -cs, c2, cs], 
        [-cs, -s2, cs, s2]])

def obtener_matriz_rigidez_portico(E, A, I, L, c, s):
    """Calcula la matriz de rigidez global 6x6 para un elemento de pórtico."""
    EA_L = E * A / L
    EIL_12 = 12 * E * I / (L**3)
    EIL_6 = 6 * E * I / (L**2)
    EIL_4 = 4 * E * I / L
    EIL_2 = 2 * E * I / L

    K_local = np.array([
        [ EA_L,   0,         0,        -EA_L,   0,         0        ],
        [ 0,      EIL_12,    EIL_6,    0,      -EIL_12,   EIL_6    ],
        [ 0,      EIL_6,     EIL_4,    0,      -EIL_6,    EIL_2    ],
        [-EA_L,   0,         0,        EA_L,    0,         0        ],
        [ 0,     -EIL_12,   -EIL_6,   0,       EIL_12,   -EIL_6   ],
        [ 0,      EIL_6,     EIL_2,    0,      -EIL_6,    EIL_4    ]
    ])

    T = np.array([
        [ c,  s,  0,  0,  0,  0 ], [-s,  c,  0,  0,  0,  0 ], [ 0,  0,  1,  0,  0,  0 ],
        [ 0,  0,  0,  c,  s,  0 ], [ 0,  0,  0, -s,  c,  0 ], [ 0,  0,  0,  0,  0,  1 ]
    ])
    
    K_global_barra = T @ K_local @ T.T
    return K_global_barra, K_local, T

def graficar_armadura_plotly(coordenadas, elementos, desplazamientos=None, escala=100):
    """
    Dibuja la armadura original y deformada (si se proveen desplazamientos)
    usando Plotly.
    
    coordenadas: ndarray (n_nodos, 2)
    elementos: list de pares de nodos [[i, j], ...]
    desplazamientos: ndarray (2*n_nodos,) o None
    escala: factor de amplificación visual para la deformada
    """
    fig = go.Figure()

    # --- Graficar la estructura original ---
    for idx, (ni, nj) in enumerate(elementos):
        xi, yi = coordenadas[ni]
        xj, yj = coordenadas[nj]
        fig.add_trace(go.Scatter(
            x=[xi, xj], y=[yi, yj],
            mode='lines+text',
            line=dict(color='black', width=3),
            name=f'Barra {idx+1}',
            text=[f'{ni+1}', f'{nj+1}'],
            textposition="top center",
            hoverinfo='text'
        ))

    # --- Graficar la deformada ---
    if desplazamientos is not None:
        n = coordenadas.shape[0]
        coords_deformadas = coordenadas + escala * desplazamientos.reshape((n, 2))

        for ni, nj in elementos:
            xi, yi = coords_deformadas[ni]
            xj, yj = coords_deformadas[nj]
            fig.add_trace(go.Scatter(
                x=[xi, xj], y=[yi, yj],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Deformada'
            ))

    fig.update_layout(
        title="Estructura: Original (negro) y Deformada (rojo)",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=False,
        autosize=True,
        height=600
    )

    return fig

# ==============================================================================
# MÓDULO PRINCIPAL DE ANÁLISIS
# ==============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import math

def analizar_estructura_streamlit():
    """
    Función principal que renderiza la interfaz de Streamlit
    y ejecuta el análisis estructural.
    """
    st.set_page_config(layout="wide", page_title="Análisis Estructural")

    st.write("Buen día Ingeniero Carlos Bravo, Cristhina Vargas le presenta:")
    st.title("🏗️ Análisis Matricial de Estructuras 2D")
    st.write("Herramienta para calcular la matriz de rigidez, desplazamientos y fuerzas internas en armaduras y pórticos planos.")

    # --- Contenedores para la interfaz ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📥 Datos de Entrada")
        # --- Valores por defecto para una estructura estable de ejemplo ---
        default_barras = [
            {"nodos_str": "1 2", "L": 640, "theta_grados": -38.66},
            {"nodos_str": "2 3", "L": 721.1, "theta_grados": 33.7},
            {"nodos_str": "1 4", "L": 500, "theta_grados": 0.0},
            {"nodos_str": "4 3", "L": 600, "theta_grados": 0.0},
            {"nodos_str": "2 4", "L": 400, "theta_grados": 90}
        ]
        default_apoyos = [
            {"nodo": 1, "tipo_str": "Articulado (Pin)"},
            {"nodo": 2, "tipo_str": "Rodillo (Roller, restringe Y)"}
        ]
        default_cargas = [
            {"nodo": 3, "fx": 200.0, "fy": -300.0, "mz": 0.0}
        ]

        # Usamos un formulario para que la app no se recargue con cada cambio
        with st.form(key="datos_form"):
            
            st.subheader("1. Tipo de Estructura")
            tipo_analisis = st.selectbox("Seleccione el tipo de análisis", ("Armadura (Cercha)", "Pórtico (Frame)"), key="tipo_analisis")
            es_portico = (tipo_analisis == "Pórtico (Frame)")
            gdl_por_nodo = 3 if es_portico else 2

            st.subheader("2. Propiedades de los Elementos")
            prop_iguales = st.radio("¿Todas las barras tienen las mismas propiedades (E, A, I)?", ("Sí, todas iguales", "No, usaré diferentes tipos"), key="prop_iguales")
            submitted = st.form_submit_button("✅ Enviar ")
            
            tipos_de_barra = []
            if prop_iguales == "Sí, todas iguales":
                E = st.number_input("Módulo de Elasticidad (E)", value=2100000, format="%.4f")
                A = st.number_input("Área (A)", value=10, format="%.4f")
                I = st.number_input("Inercia (I)", value=0.0001, format="%.5f", disabled=not es_portico) if es_portico else 0
                tipos_de_barra.append({"E": E, "A": A, "I": I})
                num_tipos = 1
            else:
                st.subheader("¿Cuántos tipos de barra diferentes usará?")
                num_tipos = st.number_input("¿Cuántos tipos de barra diferentes usará?", min_value=1, value=1, step=1)
                for i in range(num_tipos):
                    with st.expander(f"Definir Tipo de Barra {i+1}", expanded=True):
                        E = st.number_input(f"Módulo de Elasticidad (E) - Tipo {i+1}", value=2100000, format="%.4f", key=f"E_{i}")
                        A = st.number_input(f"Área (A) - Tipo {i+1}", value=10.00, format="%.4f", key=f"A_{i}")
                        I = st.number_input(f"Inercia (I) - Tipo {i+1}", value=0.0001, format="%.5f", key=f"I_{i}", disabled=not es_portico) if es_portico else 0
                        tipos_de_barra.append({"E": E, "A": A, "I": I})

            st.subheader("3. Geometría General")
            numero_nodos = st.number_input("Número total de nodos", min_value=2, value=4, step=1)
            numero_barras = st.number_input("Número total de barras", min_value=1, value=len(default_barras), step=1)

            st.subheader("4. Definición de Barras")
            definicion_elementos = []
            for i in range(numero_barras):
                default = default_barras[i] if i < len(default_barras) else {"nodos_str": f"{i} {i+1}", "L": 1.0, "theta_grados": 0.0}
                with st.expander(f"Datos para la Barra {i+1}", expanded=True):
                    nodos_barra = st.text_input(f"Nodos que conecta (ej: 1 2)", value=default["nodos_str"], key=f"nodos_{i}")
                    L = st.number_input(f"Longitud (L)", value=default["L"], key=f"L_{i}")
                    theta_grados = st.number_input(f"Ángulo (theta) en grados", value=default["theta_grados"], key=f"theta_{i}")
                    
                    tipo_asignado_idx = 0
                    if num_tipos > 1:
                        tipo_asignado_idx = st.selectbox(f"Asignar tipo de barra", options=range(num_tipos), format_func=lambda x: f"Tipo {x+1}", key=f"tipo_barra_{i}")
                    
                    definicion_elementos.append({"nodos_str": nodos_barra, "L": L, "theta_grados": theta_grados, "tipo_idx": tipo_asignado_idx})

            st.subheader("5. Cargas y Apoyos")
            num_cargas = st.number_input("¿En cuántos nodos se aplican cargas/momentos?", min_value=0, value=len(default_cargas), step=1)
            cargas_info = []
            for i in range(num_cargas):
                default = default_cargas[i] if i < len(default_cargas) else {"nodo": numero_nodos, "fx": 0.0, "fy": -1000.0, "mz": 0.0}
                with st.expander(f"Datos de Carga {i+1}", expanded=True):
                    nodo_carga = st.number_input(f"Nodo de aplicación", min_value=1, max_value=numero_nodos, value=default["nodo"], key=f"nodo_c_{i}")
                    fx = st.number_input(f"Fuerza en X (Fx)", value=default["fx"], key=f"fx_{i}")
                    fy = st.number_input(f"Fuerza en Y (Fy)", value=default["fy"], key=f"fy_{i}")
                    mz = st.number_input(f"Momento (Mz)", value=default["mz"], key=f"mz_{i}", disabled=not es_portico) if es_portico else 0
                    cargas_info.append({"nodo": nodo_carga, "fx": fx, "fy": fy, "mz": mz})

            num_apoyos = st.number_input("¿Cuántos nodos tienen apoyos?", min_value=1, value=len(default_apoyos), step=1)
            apoyos_info = []
            apoyo_opts = {"Articulado (Pin)": 1, "Empotrado (Fixed)": 2, "Rodillo (Roller, restringe Y)": 3}
            if not es_portico:
                if "Empotrado (Fixed)" in apoyo_opts: del apoyo_opts["Empotrado (Fixed)"]
            
            for i in range(num_apoyos):
                default = default_apoyos[i] if i < len(default_apoyos) else {"nodo": 1, "tipo_str": "Articulado (Pin)"}
                with st.expander(f"Datos de Apoyo {i+1}", expanded=True):
                    nodo_apoyo = st.number_input(f"Nodo con apoyo", min_value=1, max_value=numero_nodos, value=default["nodo"], key=f"nodo_a_{i}")
                    tipo_apoyo_str = st.selectbox(f"Tipo de apoyo", options=list(apoyo_opts.keys()), index=list(apoyo_opts.keys()).index(default["tipo_str"]), key=f"tipo_a_{i}")
                    apoyos_info.append({"nodo": nodo_apoyo, "tipo": apoyo_opts[tipo_apoyo_str]})

            submitted = st.form_submit_button("✅ Analizar y Calcular Matriz")

    if submitted:
        with col2:
            st.header("📊 Resultados del Análisis")
            try:
                for elem in definicion_elementos:
                    nodos = [int(n) - 1 for n in elem["nodos_str"].split()]
                    if any(n >= numero_nodos for n in nodos) or any(n < 0 for n in nodos):
                        st.error(f"Error en Barra (Nodos: {elem['nodos_str']}): El nodo especificado está fuera del rango (1 a {numero_nodos}).")
                        st.stop()
                    elem["nodos"] = nodos

                K_total_ensamblada = np.zeros((numero_nodos * gdl_por_nodo, numero_nodos * gdl_por_nodo))
                datos_calculados_barras = []

                for i, elem in enumerate(definicion_elementos):
                    tipo_barra = tipos_de_barra[elem["tipo_idx"]]
                    E, A, I = tipo_barra["E"], tipo_barra["A"], tipo_barra["I"]
                    L, theta_grados = elem["L"], elem["theta_grados"]
                    theta_rad = math.radians(theta_grados)
                    c, s = math.cos(theta_rad), math.sin(theta_rad)
                    nodo_i, nodo_j = elem["nodos"]
                    
                    if es_portico:
                        K_elemento, K_local, T_rot = obtener_matriz_rigidez_portico(E, A, I, L, c, s)
                        gdl = [3*nodo_i, 3*nodo_i+1, 3*nodo_i+2, 3*nodo_j, 3*nodo_j+1, 3*nodo_j+2]
                        datos_calculados_barras.append({'K_elemento': K_elemento, 'K_local': K_local, 'T_rot': T_rot, 'gdl': gdl})
                    else:
                        K_elemento = obtener_matriz_rigidez_armadura(E, A, L, c, s)
                        gdl = [2*nodo_i, 2*nodo_i+1, 2*nodo_j, 2*nodo_j+1]
                        datos_calculados_barras.append({'K_elemento': K_elemento, 'L': L, 'E': E, 'A': A, 'c': c, 's': s, 'gdl': gdl})
                    
                    for fila in range(gdl_por_nodo * 2):
                        for col in range(gdl_por_nodo * 2):
                            K_total_ensamblada[gdl[fila], gdl[col]] += K_elemento[fila, col]

                vector_fuerzas = np.zeros(numero_nodos * gdl_por_nodo)
                for carga in cargas_info:
                    nodo = carga["nodo"] - 1
                    if es_portico:
                        vector_fuerzas[3*nodo:3*nodo+3] += [carga["fx"], carga["fy"], carga["mz"]]
                    else:
                        vector_fuerzas[2*nodo:2*nodo+2] += [carga["fx"], carga["fy"]]

                gdl_restringidos = []
                for apoyo in apoyos_info:
                    nodo, tipo = apoyo["nodo"] - 1, apoyo["tipo"]
                    if es_portico:
                        if tipo == 1: gdl_restringidos.extend([3*nodo, 3*nodo+1])
                        elif tipo == 2: gdl_restringidos.extend([3*nodo, 3*nodo+1, 3*nodo+2])
                        elif tipo == 3: gdl_restringidos.append(3*nodo+1)
                    else:
                        if tipo == 1: gdl_restringidos.extend([2*nodo, 2*nodo+1])
                        elif tipo == 3: gdl_restringidos.append(2*nodo+1)

                gdl_totales = np.arange(numero_nodos * gdl_por_nodo)
                gdl_restringidos = sorted(list(set(gdl_restringidos)))
                gdl_libres = np.setdiff1d(gdl_totales, gdl_restringidos)
                
                K_reducida = K_total_ensamblada[np.ix_(gdl_libres, gdl_libres)]
                fuerzas_reducidas = vector_fuerzas[gdl_libres]
                
                if np.linalg.matrix_rank(K_reducida) < K_reducida.shape[0]:
                    st.error("❌ **ERROR: La estructura es inestable.** La matriz de rigidez no se puede invertir.")
                    st.warning("La matriz de rigidez reducida es singular, lo que impide la solución.")
                    st.write("Causas Comunes:")
                    st.markdown("- **Apoyos Insuficientes:** La estructura puede moverse como un cuerpo rígido.\n- **Mecanismo Interno:** Una parte de la estructura no está conectada rígidamente.\n- **Conectividad Incorrecta:** Revise la definición de nodos en las barras.")
                    st.subheader("Matriz de Rigidez Reducida (K_ff)")
                    st.dataframe(pd.DataFrame(K_reducida).style.format("{:.2e}"))
                    st.stop()

                desplazamientos_reducidos = np.linalg.solve(K_reducida, fuerzas_reducidas)
                desplazamientos = np.zeros(numero_nodos * gdl_por_nodo)
                desplazamientos[gdl_libres] = desplazamientos_reducidos

                st.subheader("Matriz de Rigidez Global Ensamblada (K)")
                st.dataframe(pd.DataFrame(K_total_ensamblada).style.format("{:.2e}"))

                # --- NUEVA SECCIÓN: MATRICES INDIVIDUALES ---
                st.subheader("Matrices de Rigidez Individuales (k)")
                for i, elem in enumerate(definicion_elementos):
                    nodos_b = elem["nodos"]
                    datos = datos_calculados_barras[i]
                    with st.expander(f"Matriz de Rigidez Global de la Barra {i+1} (Nodos {nodos_b[0]+1}-{nodos_b[1]+1})"):
                        st.dataframe(pd.DataFrame(datos['K_elemento']).style.format("{:.2e}"))

                st.subheader("Desplazamientos Nodales")
                desp_data = []
                for i in range(numero_nodos):
                    if es_portico:
                        dx, dy, rz = desplazamientos[3*i:3*i+3]
                        desp_data.append({"Nodo": i+1, "dX": dx, "dY": dy, "Rz (rad)": rz})
                    else:
                        dx, dy = desplazamientos[2*i:2*i+2]
                        desp_data.append({"Nodo": i+1, "dX": dx, "dY": dy})
                st.dataframe(pd.DataFrame(desp_data).style.format("{:.8f}"))

                st.subheader("Gráfica de la Estructura")
                coordenadas = np.zeros((numero_nodos, 2))
                # Estimamos coordenadas en base a barras si no fueron definidas (mejor si las defines directamente)
                for i, elem in enumerate(definicion_elementos):
                    n1, n2 = elem["nodos"]
                    L = elem["L"]
                    theta_rad = math.radians(elem["theta_grados"])
                    if coordenadas[n1][0] == 0 and coordenadas[n1][1] == 0:
                        coordenadas[n2][0] = coordenadas[n1][0] + L * math.cos(theta_rad)
                        coordenadas[n2][1] = coordenadas[n1][1] + L * math.sin(theta_rad)

                # Preparar conectividad
                conectividad = [elem["nodos"] for elem in definicion_elementos]

                # Reducción de desplazamientos a 2D (solo dX y dY por nodo)
                disp_reducido = desplazamientos.reshape((numero_nodos, gdl_por_nodo))[:, :2].flatten()

                # Llamar a la función de graficado
                fig = graficar_armadura_plotly(coordenadas, conectividad, disp_reducido, escala=100)
                st.plotly_chart(fig, use_container_width=True)


                st.subheader("Fuerzas Internas en los Elementos")
                for i, elem in enumerate(definicion_elementos):
                    nodos_b = elem["nodos"]
                    st.markdown(f"**Barra {i+1} (Nodos {nodos_b[0]+1}-{nodos_b[1]+1})**")
                    datos = datos_calculados_barras[i]
                    gdl = datos['gdl']
                    d_global_elem = desplazamientos[gdl]
                    
                    if es_portico:
                        K_local, T_rot = datos['K_local'], datos['T_rot']
                        d_local_elem = T_rot @ d_global_elem
                        f_local_elem = K_local @ d_local_elem
                        fuerzas_df = pd.DataFrame({'Extremo': [f"Inicial (Nodo {nodos_b[0]+1})", f"Final (Nodo {nodos_b[1]+1})"], 'Fuerza Axial': [f_local_elem[0], f_local_elem[3]], 'Fuerza Cortante': [f_local_elem[1], f_local_elem[4]], 'Momento Flector': [f_local_elem[2], f_local_elem[5]]})
                        st.dataframe(fuerzas_df.style.format("{:.3f}"))
                    else:
                        L, E, A, c, s = datos['L'], datos['E'], datos['A'], datos['c'], datos['s']
                        T_prima = np.array([-c, -s, c, s])
                        fuerza = (E * A / L) * np.dot(T_prima, d_global_elem)
                        estado = "(Tensión)" if fuerza >= 0 else "(Compresión)"
                        st.metric(label="Fuerza Axial", value=f"{fuerza:.3f}", delta=estado)
            
            except Exception as e:
                st.error(f"❌ **ERROR: Ocurrió un problema durante el cálculo.** Verifique que todos los datos de entrada sean correctos. Detalle: {e}")

if __name__ == "__main__":
    analizar_estructura_streamlit()