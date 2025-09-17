---
title: 'Building COT: How We Boosted Throughput by 30% with Microservices at Tata 1mg 💊⚡'
date: 2024-12-20
permalink: /posts/building-cot-tata1mg/
tags:
  - Microservices
  - POS-System
  - Tata-1mg
  - Async-Programming
  - Team-of-Quarter
  - E-commerce
---

Leading the development of **Common Order Taking (COT)** at Tata 1mg was one of the most challenging and rewarding experiences of my software engineering career. COT isn't just another Point of Sale (POS) system - it's the nerve center that processes thousands of medicine orders daily across India's largest online pharmacy. In this post, I'll take you through our journey of building a scalable, microservices-based POS system that achieved a **30% throughput boost** and earned our team the **Team of the Quarter** award.

## The Challenge: Scaling India's Largest Medicine Marketplace 🏥

Tata 1mg serves millions of customers across India, processing everything from prescription medicines to wellness products. When I joined the team, we faced several critical challenges:

- **Legacy monolithic system** struggling with peak loads (especially during COVID-19)
- **Inconsistent order processing** across different channels (web, mobile, partner stores)
- **Limited scalability** during promotional events and health emergencies
- **Complex inventory management** across multiple fulfillment centers
- **Integration nightmares** with various payment gateways and logistics partners

The existing system was hitting its limits, and we needed a complete architectural overhaul. Enter COT - our vision for a modern, scalable, microservices-based order taking system.

## Architecture Overview: Microservices Done Right 🏗️

We designed COT as a distributed system with clear service boundaries, each responsible for specific business capabilities:

```python
# COT System Architecture Overview
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class ServiceType(Enum):
    ORDER_MANAGEMENT = "order_management"
    INVENTORY_SERVICE = "inventory_service"
    PAYMENT_GATEWAY = "payment_gateway"
    PRESCRIPTION_VALIDATOR = "prescription_validator"
    LOGISTICS_COORDINATOR = "logistics_coordinator"
    NOTIFICATION_SERVICE = "notification_service"
    ANALYTICS_ENGINE = "analytics_engine"

@dataclass
class OrderRequest:
    customer_id: str
    items: List[Dict]
    prescription_id: Optional[str]
    delivery_address: Dict
    payment_method: str
    channel: str  # web, mobile, partner_store
    
class COTOrchestrator:
    """Main orchestrator for COT system"""
    
    def __init__(self):
        self.services = self._initialize_services()
        self.event_bus = EventBus()
        self.circuit_breakers = CircuitBreakerManager()
        
    async def process_order(self, order_request: OrderRequest) -> OrderResponse:
        """Main order processing pipeline"""
        try:
            # Step 1: Validate order
            validation_result = await self.services[ServiceType.PRESCRIPTION_VALIDATOR].validate(
                order_request
            )
            
            if not validation_result.is_valid:
                return OrderResponse(status="VALIDATION_FAILED", 
                                   errors=validation_result.errors)
            
            # Step 2: Check inventory availability
            inventory_check = await self.services[ServiceType.INVENTORY_SERVICE].check_availability(
                order_request.items
            )
            
            # Step 3: Process payment
            payment_result = await self.services[ServiceType.PAYMENT_GATEWAY].process_payment(
                order_request
            )
            
            # Step 4: Create order and coordinate fulfillment
            order_id = await self._create_order_async(order_request, payment_result)
            
            # Step 5: Trigger logistics and notifications
            await self._trigger_downstream_services(order_id, order_request)
            
            return OrderResponse(status="SUCCESS", order_id=order_id)
            
        except Exception as e:
            await self._handle_order_failure(order_request, e)
            raise
```

## Key Innovation 1: Async-First Architecture ⚡

One of the biggest performance improvements came from adopting an **async-first paradigm**. Traditional synchronous processing was a major bottleneck, especially during peak hours.

### Before: Synchronous Processing
```python
# Old synchronous approach - blocking operations
def process_order_sync(order):
    # Each step blocks the thread
    validation = validate_prescription(order)  # 200ms
    inventory = check_inventory(order.items)   # 300ms
    payment = process_payment(order)           # 500ms
    logistics = arrange_delivery(order)        # 400ms
    
    return create_order(validation, inventory, payment, logistics)
    
# Total time: ~1.4 seconds per order
# Throughput: ~43 orders/minute per worker
```

### After: Asynchronous Processing
```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class AsyncOrderProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.semaphore = asyncio.Semaphore(100)  # Limit concurrent operations
        
    async def process_order_async(self, order):
        async with self.semaphore:
            # Run independent operations concurrently
            tasks = [
                self.validate_prescription_async(order),
                self.check_inventory_async(order.items),
                self.prepare_payment_async(order)
            ]
            
            validation, inventory, payment_prep = await asyncio.gather(*tasks)
            
            # Dependent operations run sequentially
            if validation.is_valid and inventory.available:
                payment_result = await self.process_payment_async(order, payment_prep)
                logistics = await self.arrange_delivery_async(order)
                
                return await self.create_order_async(
                    validation, inventory, payment_result, logistics
                )
    
    async def validate_prescription_async(self, order):
        """Async prescription validation with external API"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{PRESCRIPTION_SERVICE_URL}/validate",
                json=order.prescription_data
            ) as response:
                return await response.json()
```

### The Results
- **Average processing time**: Reduced from 1.4s to 0.6s (57% improvement)
- **Throughput**: Increased from 43 to 167 orders/minute per worker (285% improvement)
- **Resource utilization**: 40% better CPU and memory efficiency

## Key Innovation 2: Smart Service Mesh with Circuit Breakers 🔄

With microservices, network failures become inevitable. We implemented a robust circuit breaker pattern to handle service failures gracefully:

```python
from enum import Enum
import time
import asyncio
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing - reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.failure_count = 0
            else:
                raise CircuitBreakerOpenException("Service unavailable")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

class ServiceMesh:
    def __init__(self):
        self.circuit_breakers = {}
        self.service_registry = ServiceRegistry()
        
    async def call_service(self, service_name: str, method: str, *args, **kwargs):
        """Make resilient service calls"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        
        circuit_breaker = self.circuit_breakers[service_name]
        service_instance = await self.service_registry.get_healthy_instance(service_name)
        
        return await circuit_breaker.call(
            service_instance.call_method, method, *args, **kwargs
        )
```

## Key Innovation 3: Event-Driven Order Processing 📡

We implemented an event-driven architecture that allows different services to react to order events independently:

```python
import asyncio
from typing import Dict, List, Callable
from dataclasses import dataclass
import json

@dataclass
class OrderEvent:
    event_type: str
    order_id: str
    timestamp: float
    data: Dict
    
class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_queue = asyncio.Queue()
        self.workers = []
        
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to specific event types"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event: OrderEvent):
        """Publish event to all subscribers"""
        await self.event_queue.put(event)
    
    async def start_workers(self, num_workers: int = 5):
        """Start event processing workers"""
        for i in range(num_workers):
            worker = asyncio.create_task(self._event_worker(f"worker-{i}"))
            self.workers.append(worker)
    
    async def _event_worker(self, worker_id: str):
        """Process events from queue"""
        while True:
            try:
                event = await self.event_queue.get()
                await self._process_event(event)
                self.event_queue.task_done()
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    async def _process_event(self, event: OrderEvent):
        """Process single event by notifying all subscribers"""
        if event.event_type in self.subscribers:
            tasks = []
            for handler in self.subscribers[event.event_type]:
                task = asyncio.create_task(handler(event))
                tasks.append(task)
            
            # Process all handlers concurrently
            await asyncio.gather(*tasks, return_exceptions=True)

# Example usage: Order lifecycle events
class OrderLifecycleManager:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        # Subscribe to different order events
        self.event_bus.subscribe("order.created", self.handle_order_created)
        self.event_bus.subscribe("payment.completed", self.handle_payment_completed)
        self.event_bus.subscribe("order.shipped", self.handle_order_shipped)
        
    async def handle_order_created(self, event: OrderEvent):
        """Handle new order creation"""
        order_data = event.data
        
        # Trigger inventory reservation
        await self.event_bus.publish(OrderEvent(
            event_type="inventory.reserve_request",
            order_id=event.order_id,
            timestamp=time.time(),
            data={"items": order_data["items"]}
        ))
        
        # Send confirmation SMS/email
        await self.event_bus.publish(OrderEvent(
            event_type="notification.send",
            order_id=event.order_id,
            timestamp=time.time(),
            data={
                "type": "order_confirmation",
                "customer_id": order_data["customer_id"]
            }
        ))
```

## Performance Monitoring and Observability 📊

With a distributed system, monitoring becomes crucial. We implemented comprehensive observability:

```python
import time
import logging
from prometheus_client import Counter, Histogram, Gauge
from datadog import statsd

class COTMetrics:
    def __init__(self):
        # Prometheus metrics
        self.order_counter = Counter('cot_orders_total', 'Total orders processed')
        self.order_duration = Histogram('cot_order_duration_seconds', 
                                       'Order processing duration')
        self.active_connections = Gauge('cot_active_connections', 
                                       'Active connections')
        
        # Business metrics
        self.revenue_gauge = Gauge('cot_revenue_total', 'Total revenue processed')
        self.error_rate = Counter('cot_errors_total', 'Total errors', ['error_type'])
        
    def record_order_processed(self, order_value: float, processing_time: float):
        """Record successful order processing"""
        self.order_counter.inc()
        self.order_duration.observe(processing_time)
        self.revenue_gauge.inc(order_value)
        
        # Send to DataDog for business dashboards
        statsd.increment('cot.orders.processed')
        statsd.histogram('cot.orders.value', order_value)
        statsd.timing('cot.orders.processing_time', processing_time * 1000)
    
    def record_error(self, error_type: str, order_id: str):
        """Record order processing errors"""
        self.error_rate.labels(error_type=error_type).inc()
        statsd.increment(f'cot.errors.{error_type}')
        
        logger.error(f"Order processing error: {error_type} for order {order_id}")

# Usage in order processing
class InstrumentedOrderProcessor:
    def __init__(self):
        self.metrics = COTMetrics()
        
    async def process_order(self, order_request: OrderRequest):
        start_time = time.time()
        
        try:
            result = await self._process_order_internal(order_request)
            
            processing_time = time.time() - start_time
            self.metrics.record_order_processed(
                order_value=order_request.total_value,
                processing_time=processing_time
            )
            
            return result
            
        except ValidationError as e:
            self.metrics.record_error("validation_error", order_request.id)
            raise
        except PaymentError as e:
            self.metrics.record_error("payment_error", order_request.id)
            raise
        except Exception as e:
            self.metrics.record_error("unknown_error", order_request.id)
            raise
```

## The Results: 30% Throughput Boost and Team of the Quarter 🏆

The impact of COT was transformational for Tata 1mg:

### Performance Improvements:
- **30% throughput increase**: From 2,000 to 2,600 orders/hour during peak times
- **57% reduction** in average order processing time
- **85% improvement** in system reliability (99.9% uptime vs 99.1% previously)
- **40% reduction** in infrastructure costs through better resource utilization

### Business Impact:
- **Seamless scaling** during COVID-19 demand surge (5x order volume)
- **Unified experience** across all customer touchpoints
- **Faster feature deployment** (from weeks to days)
- **Improved customer satisfaction** (4.2 to 4.7 app store rating)

### Technical Achievements:
```python
# Key metrics from production deployment
COT_PRODUCTION_METRICS = {
    "orders_per_hour_peak": 2600,  # 30% improvement
    "avg_processing_time_ms": 600,  # 57% improvement
    "system_uptime": 99.9,          # 85% improvement
    "error_rate": 0.1,              # 90% reduction
    "infrastructure_cost_reduction": 0.4,  # 40% savings
    "deployment_frequency": "daily",  # vs weekly before
    "mttr_minutes": 5,               # Mean time to recovery
    "customer_satisfaction": 4.7     # Up from 4.2
}
```

## Lessons Learned: Microservices Best Practices 📚

Building COT taught us valuable lessons about microservices architecture:

### 1. Service Boundaries Matter
```python
# Good: Domain-driven service boundaries
class OrderService:  # Handles order lifecycle
class InventoryService:  # Manages stock
class PaymentService:  # Processes payments

# Bad: Data-driven boundaries  
class UserDataService:  # Too broad
class DatabaseService:  # Infrastructure concern
```

### 2. Async Doesn't Always Mean Better
- Use async for I/O-bound operations (API calls, database queries)
- Keep CPU-intensive tasks synchronous
- Monitor async queue depths to prevent memory issues

### 3. Circuit Breakers Are Essential
- Prevent cascade failures
- Provide graceful degradation
- Monitor and alert on circuit breaker state changes

### 4. Event-Driven Architecture Scales
- Loose coupling between services
- Natural horizontal scaling
- Easier to add new features

## Team Recognition: Team of the Quarter 🌟

Our work on COT was recognized with the **Team of the Quarter** award at Tata 1mg. The recognition came not just for the technical achievement, but for:

- **Cross-functional collaboration** with product, design, and business teams
- **Zero-downtime migration** from legacy system
- **Knowledge sharing** and mentoring junior developers
- **Innovation in crisis** - scaling during the COVID-19 pandemic

## Future Enhancements and Roadmap 🚀

COT continues to evolve. Here's what we're working on:

1. **AI-Powered Demand Forecasting**: Predicting order patterns to pre-position inventory
2. **Real-time Personalization**: Dynamic pricing and recommendations
3. **Advanced Analytics**: ML-driven insights for business intelligence
4. **International Expansion**: Multi-currency and cross-border logistics

## Conclusion: Building Systems That Scale 🎯

Building COT was more than just a technical project - it was about creating a foundation for Tata 1mg's continued growth. The 30% throughput improvement wasn't just a number; it represented our ability to serve more patients, process more orders, and ultimately improve healthcare access across India.

The microservices architecture we built has become the template for other systems at Tata 1mg, proving that well-designed distributed systems can deliver both performance and maintainability.

As we continue to scale healthcare technology in India, the principles and patterns we established with COT - async-first design, event-driven architecture, comprehensive monitoring, and service resilience - remain as relevant as ever.

---

**Want to discuss microservices architecture or healthcare technology?** I'd love to connect! Reach out at [yashpathania704@gmail.com](mailto:yashpathania704@gmail.com) or find me on [LinkedIn](https://linkedin.com/in/yashhere).

*Coming up next: I'll be diving into our ML-driven auto-replenishment module for Odin and how we used ARIMA & LSTM for demand forecasting in India's largest medicine warehouse!*
