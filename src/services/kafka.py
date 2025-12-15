import asyncio
import json
import logging
import os
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.observability import get_provider

logger = logging.getLogger(__name__)
provider = get_provider()

try:
    from confluent_kafka import Consumer, KafkaException, Producer
except ImportError:  # pragma: no cover - handled at runtime
    Consumer = None
    KafkaException = Exception
    Producer = None


router_kafka_demo = APIRouter(prefix="/api/v1/kafka-demo", tags=["Kafka Demo"])


class FaultProfile(BaseModel):
    """Fault switches we can toggle from the UI."""

    latency_ms: int = Field(0, ge=0, description="Artificial delay before publishing")
    drop_probability: float = Field(
        0.0, ge=0.0, le=1.0, description="Chance to drop a produced message"
    )
    duplicate_ratio: float = Field(
        0.0, ge=0.0, le=1.0, description="Chance to duplicate a produced message"
    )
    slow_consumer_ms: int = Field(
        0, ge=0, description="Extra delay on the analytics consumer loop"
    )


class StartRequest(BaseModel):
    """Configuration for a demo run."""

    bootstrap_servers: Optional[str] = Field(
        default=None,
        description="Override Kafka bootstrap servers (defaults to env KAFKA_BOOTSTRAP_SERVERS)",
    )
    rate_per_sec: float = Field(
        5.0, gt=0, le=50, description="Messages per second from the raw producer"
    )
    max_messages: Optional[int] = Field(
        default=200, gt=0, description="Stop after this many messages (None = endless)"
    )
    scenario: str = Field(default="f1", description="Scenario label (for tagging)")
    topic_prefix: Optional[str] = Field(
        default=None,
        description="Override topic prefix (defaults to env KAFKA_TOPIC_PREFIX or 'f1')",
    )
    fault: FaultProfile = Field(default_factory=FaultProfile)


class FaultRequest(BaseModel):
    fault: FaultProfile


@dataclass
class KafkaConfig:
    bootstrap_servers: str
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    ssl_ca_location: Optional[str] = None
    topic_prefix: str = "f1"
    consumer_group: str = "f1-demo"

    def topics(self) -> Dict[str, str]:
        prefix = self.topic_prefix.strip(".")
        return {
            "raw": f"{prefix}.telemetry.raw",
            "analytics": f"{prefix}.telemetry.analytics",
            "alerts": f"{prefix}.telemetry.alerts",
        }


@dataclass
class KafkaDemoState:
    running: bool = False
    run_id: Optional[str] = None
    tasks: List[asyncio.Task] = field(default_factory=list)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    metrics: Dict[str, int] = field(default_factory=dict)
    last_message_at: Optional[str] = None
    last_error: Optional[str] = None
    fault: FaultProfile = field(default_factory=FaultProfile)
    topics: Dict[str, str] = field(default_factory=dict)
    config: Optional[KafkaConfig] = None


state = KafkaDemoState()


def require_kafka_client() -> None:
    """Guard against missing Kafka dependency."""
    if Producer is None or Consumer is None:
        raise HTTPException(
            status_code=503,
            detail="confluent-kafka is not installed. Add it to requirements and reinstall.",
        )


def _strip_comment(value: Optional[str]) -> Optional[str]:
    """Remove inline comments like 'PLAINTEXT  # comment' and trim whitespace."""
    if value is None:
        return None
    return value.split("#", 1)[0].strip()


def _sanitize_override(value: Optional[str]) -> Optional[str]:
    """Treat Swagger placeholders like 'string' or blanks as unset."""
    if value is None:
        return None
    val = str(value).strip()
    if val == "" or val.lower() in {"string", "null", "none"}:
        return None
    return val


def load_config(
    topic_override: Optional[str] = None,
    bootstrap_override: Optional[str] = None,
) -> KafkaConfig:
    """Read Kafka config from env with sane defaults."""
    bootstrap = _sanitize_override(bootstrap_override) or os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
    )
    protocol_raw = _strip_comment(os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"))
    sasl_mechanism = _strip_comment(os.getenv("KAFKA_SASL_MECHANISM"))
    sasl_username = _strip_comment(os.getenv("KAFKA_SASL_USERNAME"))
    sasl_password = _strip_comment(os.getenv("KAFKA_SASL_PASSWORD"))
    ssl_ca_location = _strip_comment(os.getenv("KAFKA_SSL_CA_LOCATION"))
    topic_prefix = _sanitize_override(topic_override) or os.getenv("KAFKA_TOPIC_PREFIX", "f1") or "f1"
    config = KafkaConfig(
        bootstrap_servers=bootstrap,
        security_protocol=protocol_raw or "PLAINTEXT",
        sasl_mechanism=sasl_mechanism,
        sasl_username=sasl_username,
        sasl_password=sasl_password,
        ssl_ca_location=ssl_ca_location,
        topic_prefix=topic_prefix,
        consumer_group=os.getenv("KAFKA_CONSUMER_GROUP", "f1-demo"),
    )
    logger.info(
        "Kafka demo config loaded: bootstrap=%s protocol=%s topic_prefix=%s consumer_group=%s",
        config.bootstrap_servers,
        config.security_protocol,
        config.topic_prefix,
        config.consumer_group,
    )
    return config


def _producer_settings(conf: KafkaConfig, client_id: str) -> Dict[str, str]:
    settings: Dict[str, str] = {
        "bootstrap.servers": conf.bootstrap_servers,
        "client.id": client_id,
        "security.protocol": conf.security_protocol,
    }

    if conf.sasl_mechanism:
        settings["sasl.mechanism"] = conf.sasl_mechanism
    if conf.sasl_username:
        settings["sasl.username"] = conf.sasl_username
    if conf.sasl_password:
        settings["sasl.password"] = conf.sasl_password
    if conf.ssl_ca_location:
        settings["ssl.ca.location"] = conf.ssl_ca_location

    return settings


def _consumer_settings(conf: KafkaConfig, client_id: str) -> Dict[str, str]:
    settings = _producer_settings(conf, client_id)
    settings.update(
        {
            "group.id": conf.consumer_group,
            "enable.auto.commit": False,
            # Use earliest so demo consumers can read messages even if producer starts first
            "auto.offset.reset": "earliest",
        }
    )
    return settings


async def _producer_loop(conf: KafkaConfig, req: StartRequest) -> None:
    """Generate synthetic F1 telemetry and publish to Kafka."""
    try:
        producer = Producer(_producer_settings(conf, client_id=f"producer-{state.run_id}"))
    except Exception as exc:  # pragma: no cover - runtime path
        state.last_error = f"Producer init failed: {exc}"
        logger.error(state.last_error)
        return
    topics = conf.topics()
    interval = 1.0 / req.rate_per_sec
    sent = 0
    logger.info(f"Kafka demo producer started for topics: {topics}")

    while not state.stop_event.is_set():
        if req.max_messages and sent >= req.max_messages:
            break

        payload = _make_f1_payload(req.scenario, sent)

        if state.fault.latency_ms > 0:
            await asyncio.sleep(state.fault.latency_ms / 1000)

        if state.fault.drop_probability and random.random() < state.fault.drop_probability:
            logger.debug("Dropping message per fault profile")
        else:
            _publish_message(producer, topics["raw"], payload)
            sent += 1
            state.metrics["produced"] = sent
            state.last_message_at = datetime.utcnow().isoformat()

            if state.fault.duplicate_ratio and random.random() < state.fault.duplicate_ratio:
                _publish_message(producer, topics["raw"], payload | {"duplicate": True})
                state.metrics["produced_duplicates"] = (
                    state.metrics.get("produced_duplicates", 0) + 1
                )

        await asyncio.sleep(interval)

    producer.flush(2)
    logger.info("Kafka demo producer stopped")


async def _consumer_analytics(conf: KafkaConfig) -> None:
    """Consume raw telemetry, enrich, and forward to analytics topic."""
    topics = conf.topics()
    try:
        consumer = Consumer(_consumer_settings(conf, client_id=f"analytics-{state.run_id}"))
    except Exception as exc:  # pragma: no cover - runtime path
        state.last_error = f"Analytics consumer init failed: {exc}"
        logger.error(state.last_error)
        return
    consumer.subscribe([topics["raw"]])
    try:
        producer = Producer(_producer_settings(conf, client_id=f"analytics-p-{state.run_id}"))
    except Exception as exc:  # pragma: no cover - runtime path
        state.last_error = f"Analytics producer init failed: {exc}"
        logger.error(state.last_error)
        consumer.close()
        return
    logger.info(f"Kafka analytics consumer subscribed to {topics['raw']}")

    try:
        while not state.stop_event.is_set():
            msg = await asyncio.to_thread(consumer.poll, 1.0)
            if msg is None:
                continue
            if msg.error():
                err = str(msg.error())
                state.last_error = f"analytics consumer error: {err}"
                logger.error(state.last_error)
                continue

            if msg.headers():
                logger.debug(f"Kafka message headers: {msg.headers()}")
            else:
                logger.debug("Kafka message has no headers")

            data = json.loads(msg.value().decode("utf-8"))
            enriched = _enrich_payload(data)
            _publish_message(producer, topics["analytics"], enriched)
            state.metrics["analytics_consumed"] = state.metrics.get(
                "analytics_consumed", 0
            ) + 1

            consumer.commit(message=msg, asynchronous=True)
            if state.fault.slow_consumer_ms:
                await asyncio.sleep(state.fault.slow_consumer_ms / 1000)
    except KafkaException as exc:  # pragma: no cover - runtime path
        state.last_error = str(exc)
        logger.error(f"Kafka analytics consumer exception: {exc}")
    finally:
        consumer.close()
        producer.flush(2)
        logger.info("Kafka analytics consumer stopped")


async def _consumer_alerts(conf: KafkaConfig) -> None:
    """Consume analytics events and emit alerts when thresholds trip."""
    topics = conf.topics()
    try:
        consumer = Consumer(_consumer_settings(conf, client_id=f"alerts-{state.run_id}"))
    except Exception as exc:  # pragma: no cover - runtime path
        state.last_error = f"Alerts consumer init failed: {exc}"
        logger.error(state.last_error)
        return
    consumer.subscribe([topics["analytics"]])
    try:
        producer = Producer(_producer_settings(conf, client_id=f"alerts-p-{state.run_id}"))
    except Exception as exc:  # pragma: no cover - runtime path
        state.last_error = f"Alerts producer init failed: {exc}"
        logger.error(state.last_error)
        consumer.close()
        return
    logger.info(f"Kafka alert consumer subscribed to {topics['analytics']}")

    try:
        while not state.stop_event.is_set():
            msg = await asyncio.to_thread(consumer.poll, 1.0)
            if msg is None:
                continue
            if msg.error():
                err = str(msg.error())
                state.last_error = f"alerts consumer error: {err}"
                logger.error(state.last_error)
                continue

            data = json.loads(msg.value().decode("utf-8"))
            if _should_alert(data):
                alert = {
                    "event": "alert",
                    "run_id": state.run_id,
                    "car_id": data.get("car_id"),
                    "lap": data.get("lap"),
                    "reason": "degradation_high",
                    "degradation_score": data.get("degradation_score"),
                    "battery_soc": data.get("battery_soc"),
                    "ers_mode": data.get("ers_mode"),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                _publish_message(producer, topics["alerts"], alert)
                state.metrics["alerts_emitted"] = state.metrics.get("alerts_emitted", 0) + 1

            state.metrics["alerts_consumed"] = state.metrics.get("alerts_consumed", 0) + 1
            consumer.commit(message=msg, asynchronous=True)
    except KafkaException as exc:  # pragma: no cover - runtime path
        state.last_error = str(exc)
        logger.error(f"Kafka alert consumer exception: {exc}")
    finally:
        consumer.close()
        producer.flush(2)
        logger.info("Kafka alert consumer stopped")


def _publish_message(producer: Producer, topic: str, payload: Dict) -> None:
    """Publish and tag with observability spans."""
    body = json.dumps(payload).encode("utf-8")
    with provider.trace_context("kafka.demo.produce", resource="kafka.produce") as span:
        if span:
            span.set_tag("kafka.topic", topic)
            span.set_tag("demo.run_id", state.run_id or "unknown")
        producer.produce(topic, value=body)
        producer.poll(0)


def _make_f1_payload(scenario: str, sequence: int) -> Dict:
    """Return a synthetic telemetry payload."""
    battery_soc = round(random.uniform(0.35, 0.95), 3)
    tire_temps = [round(random.uniform(70, 105), 1) for _ in range(4)]
    payload = {
        "scenario": scenario,
        "car_id": f"CAR-{random.randint(1, 4)}",
        "lap": random.randint(1, 60),
        "sequence": sequence,
        "timestamp": datetime.utcnow().isoformat(),
        "speed_kph": round(random.uniform(90, 320), 1),
        "gear": random.randint(1, 8),
        "throttle_pct": round(random.uniform(0.3, 1.0), 2),
        "brake_pct": round(random.uniform(0, 1.0), 2),
        "tire_temp_c": tire_temps,
        "battery_soc": battery_soc,
        "ers_mode": random.choice(["deploy", "harvest", "balanced"]),
        "track_segment": random.choice(["straight", "chicane", "hairpin"]),
    }
    return payload


def _enrich_payload(data: Dict) -> Dict:
    """Derive simple degradation / fault scores."""
    temps = data.get("tire_temp_c", [])
    avg_temp = sum(temps) / len(temps) if temps else 0
    degradation_score = min(1.0, max(0.0, (avg_temp - 80) / 35))
    battery_soc = data.get("battery_soc", 0.5)
    ers_mode = data.get("ers_mode", "balanced")
    fault_flags = []

    if degradation_score > 0.65:
        fault_flags.append("tire_wear")
    if battery_soc < 0.4 and ers_mode == "deploy":
        fault_flags.append("low_battery_on_deploy")

    enriched = {
        **data,
        "degradation_score": round(degradation_score, 3),
        "fault_flags": fault_flags,
        "computed_at": datetime.utcnow().isoformat(),
    }
    return enriched


def _should_alert(data: Dict) -> bool:
    return bool(
        data.get("degradation_score", 0) > 0.7
        or data.get("fault_flags")
    )


async def _stop_all_tasks() -> None:
    """Signal stop, wait for tasks, and reset state."""
    state.stop_event.set()
    for task in state.tasks:
        if not task.done():
            task.cancel()

    if state.tasks:
        await asyncio.gather(*state.tasks, return_exceptions=True)

    state.running = False
    state.tasks = []
    state.run_id = None
    state.config = None
    state.topics = {}
    logger.info("Kafka demo tasks stopped")


@router_kafka_demo.post("/start")
async def start_kafka_demo(req: StartRequest):
    """Start the F1 Kafka demo."""
    require_kafka_client()

    if state.running:
        return await kafka_demo_status()

    conf = load_config(req.topic_prefix, req.bootstrap_servers)
    state.config = conf
    state.topics = conf.topics()
    state.run_id = str(uuid.uuid4())
    state.metrics = {"produced": 0, "analytics_consumed": 0, "alerts_consumed": 0}
    state.fault = req.fault
    state.stop_event = asyncio.Event()
    state.last_error = None

    state.tasks = [
        asyncio.create_task(_producer_loop(conf, req)),
        asyncio.create_task(_consumer_analytics(conf)),
        asyncio.create_task(_consumer_alerts(conf)),
    ]
    state.running = True
    logger.info(
        f"Kafka demo started run_id={state.run_id} bootstrap={conf.bootstrap_servers} "
        f"topics={state.topics} rate={req.rate_per_sec}"
    )
    return await kafka_demo_status()


@router_kafka_demo.post("/stop")
async def stop_kafka_demo():
    """Stop all Kafka demo tasks."""
    if not state.running:
        return {"running": False, "message": "Kafka demo is not running"}

    await _stop_all_tasks()
    return {"running": False, "message": "Kafka demo stopped"}


@router_kafka_demo.get("/status")
async def kafka_demo_status():
    """Return the current status for the UI."""
    return {
        "running": state.running,
        "run_id": state.run_id,
        "topics": state.topics,
        "bootstrap_servers": state.config.bootstrap_servers if state.config else None,
        "metrics": state.metrics,
        "last_message_at": state.last_message_at,
        "fault": state.fault,
        "last_error": state.last_error,
    }


@router_kafka_demo.post("/fault")
async def update_fault_profile(req: FaultRequest):
    """Update the fault profile on the fly."""
    state.fault = req.fault
    logger.info(f"Kafka fault profile updated: {req.fault}")
    return await kafka_demo_status()
