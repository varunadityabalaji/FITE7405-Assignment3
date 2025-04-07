from math import sqrt, log,exp,pi
from scipy.stats import norm

def calculate_d1_d2(S,K,T,t,sigma,r, q=0):
    d1= ((log(S/K) + (r-q)*(T-t))/(sigma*sqrt(T-t)))+0.5*sigma*sqrt(T-t)
    d2=d1-(sigma*sqrt(T-t))
    return d1, d2

def call(S,K,T,t,sigma,r, q=0):
    d1, d2 = calculate_d1_d2(S,K,T,t,sigma,r)
    C=S*norm.cdf(d1)*exp(-1*q*(T-t))- K*exp(-1*r*(T-t))*norm.cdf(d2)
    return C

def put(S,K,T,t,sigma,r, q=0):
    d1, d2 = calculate_d1_d2(S,K,T,t,sigma,r)
    P=K*exp(-1*r*(T-t))*norm.cdf(-1*d2) - S*norm.cdf(-d1)*exp(-1*q*(T-t))
    return P
  
def vega(S,K,T,t,r,sigma,q):
    d1,d2=calculate_d1_d2(S,K,T,t,sigma,r,q)
    vega= S*exp(-1*q*(T-t))*sqrt(T-t)*exp(-1*0.5*d1*sigma**2)/sqrt(2*pi)
    return vega

def verify_bounds(S,K,T,r,q,price,optionType):
    if optionType=='C':
        call_lower_bound = max(S*exp(-1*q*T)-K*exp(-1*r*T),0)
        call_upper_bound = S*exp(-1*q*T)
        if price >= call_lower_bound and price < call_upper_bound:
            return True
        else:
            return False
        
    if optionType=='P':
        put_lower_bound = max(K*exp(-1*r*T)-S*exp(-1*q*T), 0) 
        put_upper_bound = K*exp(-1*r*T)
        if price>= put_lower_bound and price < put_upper_bound:
            return True
        else:
            return False

def implied_volatility(S,K,T,t,r,q, Ctrue, OptionType='C'):
  sigmahat = sqrt(2*abs( (log(S/K) + (r-q)*(T-t))/(T-t) ) ) # from question and the lecture notes
  tol = 1e-8; # Tolerance
  nmax = 1000  
  sigmadiff=1
  n=1
  sigma=sigmahat
  
  while (sigmadiff>=tol and nmax>n):
      
      if OptionType == 'C':
        C = call(S,K,T,t,sigma,r,q)
      else:
         C = put(S,K,T,t,sigma,r,q)
         
      Cvega = vega(S,K,T,t,r,sigma,q)
      increment= (C-Ctrue)/Cvega
      sigma=sigma-increment
      n=n+1
      sigmadiff=abs(increment)
  
  return sigma