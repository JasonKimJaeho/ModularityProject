import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from sklearn.inspection import permutation_importance
from collections import defaultdict


def calculate_metrics(y_true, y_pred):
    # RMSE (Root Mean Squared Error) 계산
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAE (Mean Absolute Error) 계산
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error) 계산
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    
    # y_true와 y_pred 간의 상관계수 계산
    corr = np.corrcoef(y_true.T, y_pred.T)[0, 1]
    
    # 예측 값의 부호가 실제 값의 부호와 일치하는 비율 계산
    sign_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    # 실제 값과 예측 값의 방향 정확도 계산
    directional_accuracy = np.mean((np.diff(y_true, axis=0) > 0) == (np.diff(y_pred, axis=0) > 0))
    
    # 계산된 메트릭 반환
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Correlation': corr,
        'Sign Accuracy': sign_accuracy,
        'Directional Accuracy': directional_accuracy
    }

# 모델별로 특성 중요도 평가 함수 정의
def get_feature_importances(model, X_train, y_train):
    result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1)
    return result.importances_mean


import numpy as np
from scipy.optimize import minimize

# 1. 평균-분산 최적화 모형
def mean_variance_portfolio(mean_returns, cov_matrix, target_return):
    """
    평균-분산 최적화 모형을 사용하여 주어진 목표 수익률을 달성하는 포트폴리오의 비중을 계산합니다.
    
    Parameters:
    mean_returns (array): 자산의 기대 수익률
    cov_matrix (array): 자산 수익률의 공분산 행렬
    target_return (float): 목표 수익률
    
    Returns:
    array: 최적화된 포트폴리오 비중
    """
    num_assets = len(mean_returns)
    
    # 포트폴리오 분산 계산 함수
    def portfolio_variance(weights, cov_matrix):
        return weights.T @ cov_matrix @ weights

    # 포트폴리오 수익률 계산 함수
    def portfolio_return(weights, mean_returns):
        return weights.T @ mean_returns

    # 제약 조건: 비중의 합은 1이어야 함
    def constraint_sum_of_weights(weights):
        return np.sum(weights) - 1

    # 제약 조건: 포트폴리오 수익률은 목표 수익률 이상이어야 함
    def constraint_min_return(weights, mean_returns, target_return):
        return portfolio_return(weights, mean_returns) - target_return

    # 초기값 설정 (균등하게 분배)
    init_guess = np.array([1.0 / num_assets] * num_assets)

    # 제약 조건을 딕셔너리 형태로 정의
    constraints = (
        {'type': 'eq', 'fun': constraint_sum_of_weights},
        {'type': 'ineq', 'fun': lambda weights: constraint_min_return(weights, mean_returns, target_return)}
    )

    # 경계 조건 설정 (각 비중은 0 이상)
    bounds = tuple((0, 1) for _ in range(num_assets))

    # 최적화 실행
    optimal_portfolio = minimize(
        portfolio_variance,
        init_guess,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return optimal_portfolio.x * 100


# 2. 최소 분산 모형
def minimum_variance_portfolio(cov_matrix):
    """
    최소 분산 모형을 사용하여 포트폴리오의 최소 분산을 달성하는 비중을 계산합니다.
    
    Parameters:
    cov_matrix (array): 자산 수익률의 공분산 행렬
    
    Returns:
    array: 최적화된 포트폴리오 비중
    """
    num_assets = len(cov_matrix)
    
    # 목적 함수: 포트폴리오 분산
    def objective(weights, cov_matrix):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    # 제약 조건: 비중의 합은 1이어야 함
    constraints = ({
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    })
    
    # 경계 조건 설정 (각 비중은 0 이상)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # 초기값 설정 (균등하게 분배)
    init_guess = np.array(num_assets * [1. / num_assets])
    
    # 최적화 실행
    optimal_portfolio = minimize(objective, init_guess, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)
    
    return optimal_portfolio.x * 100

# 3. 위험 동등 모형
def risk_parity_portfolio(cov_matrix):
    """
    위험 동등 모형을 사용하여 각 자산이 포트폴리오의 총 위험에 동일하게 기여하도록 하는 비중을 계산합니다.
    
    Parameters:
    cov_matrix (array): 자산 수익률의 공분산 행렬
    
    Returns:
    array: 최적화된 포트폴리오 비중
    """
    num_assets = len(cov_matrix)
    
    # 포트폴리오 분산 계산 함수
    def portfolio_variance(weights, cov_matrix):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    # 한계 위험 기여도 계산 함수
    def marginal_risk_contribution(weights, cov_matrix):
        return np.dot(cov_matrix, weights)

    # 위험 기여도 계산 함수
    def risk_contribution(weights, cov_matrix):
        portfolio_variance_value = portfolio_variance(weights, cov_matrix)
        mrc = marginal_risk_contribution(weights, cov_matrix)
        return weights * mrc, portfolio_variance_value

    # 목적 함수: 각 자산의 위험 기여도의 차이를 최소화
    def objective(weights, cov_matrix):
        rc, portfolio_variance_value = risk_contribution(weights, cov_matrix)
        target_rc = portfolio_variance_value / num_assets
        return np.sum((rc - target_rc)**2)

    # 제약 조건: 비중의 합은 1이어야 함
    constraints = ({
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    })
    
    # 경계 조건 설정 (각 비중은 0 이상)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # 초기값 설정 (균등하게 분배)
    init_guess = np.array(num_assets * [1. / num_assets])
    
    # 최적화 실행
    optimal_portfolio = minimize(objective, init_guess, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)
    
    return optimal_portfolio.x * 100

# 4. 동일 비중 모형
def equal_weighting_portfolio(num_assets):
    """
    동일 비중 모형을 사용하여 각 자산에 동일한 비중을 부여합니다.
    
    Parameters:
    num_assets (int): 자산의 수
    
    Returns:
    array: 동일 비중 포트폴리오 비중
    """
    return np.array(num_assets * [1. / num_assets]) * 100

def create_scenarios(df):
    normal = df.mean()
    percentile_20 = df.quantile(0.25)
    percentile_80 = df.quantile(0.75)
    
    scenarios = pd.DataFrame({
        'Normal': normal,
        'Weak': percentile_80,
        'Robust': percentile_20
    })
    
    return scenarios

def mahalanobis_distance(instance, data):
    """
    마할라노비스 거리를 계산하는 함수

    Parameters:
    instance (array): 시나리오를 구성하는 경제변수 값
    data (array): 경제변수의 역사적 데이터

    Returns:
    float: 마할라노비스 거리
    """
    mean = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    diff = instance - mean
    distance = np.dot(np.dot(diff.T, inv_cov_matrix), diff)
    return distance

def calculate_probabilities(distances):
    """
    마할라노비스 거리를 기반으로 시나리오 발생 가능성을 계산하는 함수

    Parameters:
    distances (list): 각 시나리오의 마할라노비스 거리

    Returns:
    list: 각 시나리오의 발생 가능성
    """
    exp_distances = [np.exp(-d / 2) for d in distances]
    total = sum(exp_distances)
    probabilities = [d * 100 / total for d in exp_distances]
    return probabilities

