
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Literal, List, Dict, Any
from geometric_asian_option import geometric_asian_option_price
from american_option import american_option_pricing
from geometric_basket_asian_option import geometric_basket_asian_option_price
import os
from dotenv import load_dotenv
from monte_carlo_asian_option import AsianOptionPricer
from monte_carlo_basket import BasketOptionPricer
from quasi_kiko_option import KIKOOptionPricerQMC
from implied_vol import implied_volatility, call, put

load_dotenv()

app = FastAPI(title="Option Pricing Chatbot API")

# Data models for request validation
class GeometricAsianRequest(BaseModel):
    S0: float
    sigma: float
    r: float
    T: float
    K: float
    n: int
    option_type: Literal['call', 'put']

class AmericanRequest(BaseModel):
    S0: float
    K: float
    r: float
    T: float
    sigma: float
    N: int
    option_type: Literal['call', 'put']

class GeometricBasketAsianRequest(BaseModel):
    S1_0: float
    S2_0: float
    sigma1: float
    sigma2: float
    r: float
    T: float
    K: float
    rho: float
    type: Literal['call', 'put']

class ChatRequest(BaseModel):
    message: str

class AsianOptionRequest(BaseModel):
    S: float
    k: float
    sigma: float
    r: float
    T: float
    N: int
    M: int
    option_type: Literal['call', 'put']
    control_variate: bool

class BasketOptionRequest(BaseModel):
    S1: float
    S2: float
    K: float
    T: float
    r: float
    sigma1: float
    sigma2: float
    rho: float
    M: int
    option_type: Literal['call', 'put']
    use_control_variate: bool

class KIKOOptionRequest(BaseModel):
    S: float
    K: float
    T: float
    r: float
    sigma: float
    L: float
    U: float
    N: int
    option_type: Literal['call', 'put']
    use_control_variate: bool

class ImpliedVolatilityRequest(BaseModel):
    S: float
    K: float
    T: float
    t: float
    r: float
    q: float
    Ctrue: float
    OptionType: Literal['C', 'P']

class BlackScholesRequest(BaseModel):
    S: float
    K: float
    T: float
    r: float
    q: float
    OptionType: Literal['C', 'P']

# Option pricing endpoints
@app.get("/price/geometric-asian")
def price_geometric_asian(request: GeometricAsianRequest):
    try:
        price = geometric_asian_option_price(
            request.S0, request.sigma, request.r, request.T,
            request.K, request.n, request.option_type
        )
        return {"price": float(price)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/price/american")
def price_american(request: AmericanRequest):
    try:
        price = american_option_pricing(
            request.S0, request.K, request.r, request.T,
            request.sigma, request.N, request.option_type
        )
        return {"price": float(price)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/price/geometric-basket-asian")
def price_geometric_basket_asian(request: GeometricBasketAsianRequest):
    try:
        price = geometric_basket_asian_option_price(
            request.S1_0, request.S2_0, request.sigma1,
            request.sigma2, request.r, request.T,
            request.K, request.rho, request.type
        )
        return {"price": float(price)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/price/monte-carlo-asian")  
def price_monte_carlo_asian(request: AsianOptionRequest):
    try:
        pricer = AsianOptionPricer(
            S=request.S, k=request.k, sigma=request.sigma, r=request.r, T=request.T,
            N=request.N, M=request.M, option_type=request.option_type, control_variate=request.control_variate
        )
        result = pricer.monte_carlo_pricing()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.get("/price/monte-carlo-basket")
def price_monte_carlo_basket(request: BasketOptionRequest):
    try:
        pricer = BasketOptionPricer(
            S1=request.S1, S2=request.S2, K=request.K, T=request.T,
            r=request.r, sigma1=request.sigma1, sigma2=request.sigma2, rho=request.rho, 
            M=request.M, option_type=request.option_type, use_control_variate=request.use_control_variate
        )
        result = pricer.monte_carlo_pricing()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.get("/price/quasi-kiko")
def price_quasi_kiko(request: KIKOOptionRequest):
    try:
        pricer = KIKOOptionPricerQMC(
            S=request.S, K=request.K, T=request.T, r=request.r, sigma=request.sigma, 
            L=request.L, U=request.U, N=request.N, option_type=request.option_type, use_control_variate=request.use_control_variate
        )
        result = pricer.monte_carlo_pricing()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/implied-volatility")
def implied_volatility_endpoint(request: ImpliedVolatilityRequest):
    try:
        result = implied_volatility(
            S=request.S, K=request.K, T=request.T, t=request.t, r=request.r, q=request.q,
            Ctrue=request.Ctrue, OptionType=request.OptionType
        )
        return {"implied_volatility": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/price/black-scholes")
def price_black_scholes(request: BlackScholesRequest):
    try:
        if request.OptionType == 'C':
            price = call(request.S, request.K, request.T, request.t, request.r, request.q)
        else:
            price = put(request.S, request.K, request.T, request.t, request.r, request.q)
        return {"price": float(price)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Initialize LangChain tools and agent
# def init_agent():
#     llm = ChatOpenAI(temperature=0)
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
#     tools = [
#         Tool(
#             name="GeometricAsianOption",
#             func=lambda x: price_geometric_asian(GeometricAsianRequest(**eval(x))),
#             description="Prices a geometric Asian option. Input should be a dictionary with keys: S0 (spot price), sigma (volatility), r (interest rate), T (time to maturity), K (strike price), n (number of observations), option_type ('call' or 'put')"
#         ),
#         Tool(
#             name="AmericanOption",
#             func=lambda x: price_american(AmericanRequest(**eval(x))),
#             description="Prices an American option. Input should be a dictionary with keys: S0 (spot price), K (strike price), r (interest rate), T (time to maturity), sigma (volatility), N (number of steps), option_type ('call' or 'put')"
#         ),
#         Tool(
#             name="GeometricBasketAsianOption",
#             func=lambda x: price_geometric_basket_asian(GeometricBasketAsianRequest(**eval(x))),
#             description="Prices a geometric basket Asian option. Input should be a dictionary with keys: S1_0 (first spot price), S2_0 (second spot price), sigma1 (first volatility), sigma2 (second volatility), r (interest rate), T (time to maturity), K (strike price), rho (correlation), type ('call' or 'put')"
#         )
#     ]
    
#     return initialize_agent(
#         tools,
#         llm,
#         agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
#         verbose=True,
#         memory=memory
#     )

# agent = init_agent()

# @app.post("/chat")
# async def chat_endpoint(request: ChatRequest):
#     try:
#         response = agent.run(request.message)
#         return {"response": response}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
