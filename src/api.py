import pandas as pd
import numpy as np
import cvxpy as cp
from flask import Flask, request, jsonify, render_template_string
import os
import tempfile
import logging
import json

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Límite de 16MB para archivos

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def black_litterman_cvar_optimization(returns, risk_level, max_weight=1.0, risk_aversion=2.5, tau=0.05, view_confidence=None, market_views=None):
    """
    Implementa el modelo Black-Litterman con restricciones de CVaR.
    
    Args:
        returns (pd.DataFrame): DataFrame con los retornos diarios
        risk_level (float): Nivel máximo de riesgo permitido (CVaR)
        max_weight (float): Peso máximo permitido por activo
        risk_aversion (float): Coeficiente de aversión al riesgo
        tau (float): Parámetro de incertidumbre sobre el equilibrio
        view_confidence (dict): Confianza en las visiones (opcional)
        market_views (dict): Visiones del mercado como {ticker: expected_return} (opcional)
    
    Returns:
        dict: Diccionario con los pesos óptimos por ticker
    """
    n_assets = returns.shape[1]
    assets = returns.columns.tolist()
    
    # 1. Calcular matriz de covarianza
    cov_matrix = returns.cov().values
    
    # 2. Estimar retornos de equilibrio (usando un proxy del portafolio de mercado)
    # Para simplificar, usamos pesos iguales como proxy inicial del mercado
    mkt_weights = np.ones(n_assets) / n_assets
    
    # Retornos implícitos de equilibrio
    implied_returns = risk_aversion * cov_matrix @ mkt_weights
    
    # 3. Incorporar visiones del mercado si existen
    if market_views is not None:
        # Matriz de selección P (cada fila corresponde a una visión)
        views_indices = [assets.index(ticker) for ticker in market_views.keys()]
        n_views = len(views_indices)
        P = np.zeros((n_views, n_assets))
        q = np.zeros(n_views)
        
        for i, idx in enumerate(views_indices):
            P[i, idx] = 1
            ticker = assets[idx]
            q[i] = market_views[ticker]
        
        # Matriz de covarianza de las visiones (incertidumbre)
        if view_confidence is None:
            # Nivel de confianza por defecto
            omega = np.diag(np.diag(P @ cov_matrix @ P.T)) * tau
        else:
            # Ajustar por confianza personalizada
            confidence_values = np.array([view_confidence.get(assets[idx], 1.0) for idx in views_indices])
            omega = np.diag(np.diag(P @ cov_matrix @ P.T)) * tau / confidence_values[:, np.newaxis]
        
        # Fórmula Black-Litterman
        BL_term1 = np.linalg.inv(tau * cov_matrix)
        BL_term2 = P.T @ np.linalg.inv(omega) @ P
        BL_term3 = np.linalg.inv(tau * cov_matrix) @ implied_returns
        BL_term4 = P.T @ np.linalg.inv(omega) @ q
        
        BL_posterior_cov = np.linalg.inv(BL_term1 + BL_term2)
        BL_posterior_returns = BL_posterior_cov @ (BL_term3 + BL_term4)
    else:
        # Si no hay visiones, usamos los retornos implícitos
        BL_posterior_returns = implied_returns
        BL_posterior_cov = cov_matrix
    
    # 4. Optimización con CVaR
    weights = cp.Variable(n_assets)
    
    # Parámetros para CVaR
    alpha = 0.95  # Nivel de confianza para CVaR
    n_scenarios = returns.shape[0]
    
    # Variables auxiliares para CVaR
    var = cp.Variable(1)
    aux = cp.Variable(n_scenarios)
    
    # Función objetivo: maximizar el retorno esperado con los retornos de Black-Litterman
    objective = cp.Maximize(BL_posterior_returns @ weights)
    
    # Restricciones
    constraints = [
        cp.sum(weights) == 1,              # Suma de pesos = 1
        weights >= 0,                      # No short-selling
        weights <= max_weight,             # Límite máximo por activo
    ]
    
    # Restricciones para CVaR
    for i in range(n_scenarios):
        scenario_return = returns.iloc[i].values
        constraints.append(aux[i] >= -scenario_return @ weights - var)
    
    constraints.append(aux >= 0)
    constraints.append(var + (1/(1-alpha)) * (1/n_scenarios) * cp.sum(aux) <= risk_level)
    
    # Resolver el problema
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"El problema no se resolvió de manera óptima. Estado: {problem.status}")
            return None
        
        # Convertir los pesos en un diccionario
        optimal_weights = weights.value
        optimal_portfolio = {assets[i]: round(float(optimal_weights[i]), 4) for i in range(n_assets) if optimal_weights[i] > 1e-4}
        
        return optimal_portfolio
        
    except cp.error.SolverError as e:
        logger.error(f"Error en el solver: {e}")
        return None
    except Exception as e:
        logger.error(f"Error durante la optimización: {e}")
        return None

@app.route('/optimize-portfolio', methods=['POST'])
def optimize():
    """
    Endpoint para optimizar un portafolio basado en retornos históricos
    usando el modelo Black-Litterman con restricciones de CVaR.
    
    Inputs:
        - Archivo CSV con retornos diarios (formato ticker/fecha)
        - risk_level: nivel máximo de riesgo permitido (CVaR)
        - max_weight: peso máximo permitido por activo
        - risk_aversion: coeficiente de aversión al riesgo (opcional)
        - market_views: visiones de mercado en formato JSON (opcional)
    
    Returns:
        JSON con el portafolio óptimo
    """
    try:
        # Verificar si se proporciona un archivo
        if 'file' not in request.files:
            return jsonify({"error": "No se proporcionó un archivo CSV"}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No se seleccionó un archivo"}), 400
            
        if not allowed_file(file.filename):
            return jsonify({"error": "Formato de archivo no permitido. Solo se aceptan archivos CSV"}), 400
        
        # Obtener parámetros
        risk_level = float(request.form.get('risk_level', 0.02))
        max_weight = float(request.form.get('max_weight', 0.2))
        risk_aversion = float(request.form.get('risk_aversion', 2.5))
        
        # Obtener visiones de mercado si existen
        market_views = None
        if 'market_views' in request.form:
            try:
                market_views = json.loads(request.form.get('market_views'))
            except Exception as e:
                return jsonify({"error": f"Error en el formato de market_views: {str(e)}"}), 400
        
        # Obtener confianza en las visiones si existe
        view_confidence = None
        if 'view_confidence' in request.form:
            try:
                view_confidence = json.loads(request.form.get('view_confidence'))
            except Exception as e:
                return jsonify({"error": f"Error en el formato de view_confidence: {str(e)}"}), 400
        
        # Leer el archivo CSV
        try:
            # Guardar el archivo temporalmente
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                file.save(tmp.name)
                returns = pd.read_csv(tmp.name, index_col=0, parse_dates=True)
            os.unlink(tmp.name)  # Eliminar el archivo temporal
        except Exception as e:
            return jsonify({"error": f"Error al leer el archivo CSV: {str(e)}"}), 400
        
        # Validar datos
        if returns.isnull().values.any():
            return jsonify({"error": "El CSV contiene valores nulos"}), 400
            
        # Optimizar el portafolio usando Black-Litterman con CVaR
        optimal_portfolio = black_litterman_cvar_optimization(
            returns=returns,
            risk_level=risk_level,
            max_weight=max_weight,
            risk_aversion=risk_aversion,
            view_confidence=view_confidence,
            market_views=market_views
        )
        
        if optimal_portfolio is None:
            return jsonify({"error": "No se pudo encontrar un portafolio optimo con los parametros proporcionados"}), 400
            
        # Devolver el resultado
        return jsonify({"optimal_portfolio": optimal_portfolio})
        
    except Exception as e:
        logger.error(f"Error en el endpoint: {e}")
        return jsonify({"error": f"Error en el servicio: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def index():
    response_format = {
        "optimal_portfolio": {
            "ticker_1": "weight_1",
            "ticker_2": "weight_2",
            "...": "..."
        }
    }
    json_string = json.dumps(response_format)
    html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ISD Fintual Test</title>
        </head>
        <body>
            <h1>API de Optimización de Portafolios con Black-Litterman y CVaR</h1>
            <h3>
                Servicio: /optimize-portfolio
            </h3>
            <h4>
                Parámetros
            </h4>
            <ul>
                <li>
                    file": "Archivo CSV con retornos diarios (formato ticker/fecha)
                </li>
                <li>
                    risk_level": "Nivel máximo de riesgo permitido (CVaR)
                </li>
                <li>
                    max_weight": "Peso máximo permitido por activo
                </li>
            </ul>
            <h4>
                Formato de respuesta
            </h4>
            <p>
                <pre id="json-data">{json_string}</pre>
            </p>
        </body>
        </html>
    """
    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)