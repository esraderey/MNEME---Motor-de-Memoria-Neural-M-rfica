"""
MNEME Security Module
Módulo de seguridad avanzado con verificación criptográfica, árboles Merkle y auditoría
"""

import hashlib
import hmac
import secrets
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import pickle
import struct

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Niveles de seguridad"""
    BASIC = 1
    STANDARD = 2
    HIGH = 3
    MAXIMUM = 4

class AuditEvent(Enum):
    """Tipos de eventos de auditoría"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    VERIFY = "verify"
    COMPRESS = "compress"
    DECOMPRESS = "decompress"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    SECURITY_VIOLATION = "security_violation"

@dataclass
class AuditLog:
    """Entrada de auditoría"""
    timestamp: float
    event: AuditEvent
    resource_id: str
    user_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        return {
            "timestamp": self.timestamp,
            "event": self.event.value,
            "resource_id": self.resource_id,
            "user_id": self.user_id,
            "details": self.details,
            "signature": self.signature.hex() if self.signature else None
        }

class CryptographicVerifier:
    """Verificador criptográfico avanzado"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.security_level = security_level
        self.key = self._generate_key()
        
    def _generate_key(self) -> bytes:
        """Generar clave criptográfica"""
        if self.security_level == SecurityLevel.BASIC:
            return secrets.token_bytes(16)
        elif self.security_level == SecurityLevel.STANDARD:
            return secrets.token_bytes(32)
        elif self.security_level == SecurityLevel.HIGH:
            return secrets.token_bytes(64)
        else:  # MAXIMUM
            return secrets.token_bytes(128)
    
    def compute_checksum(self, data: bytes, algorithm: str = "sha256") -> bytes:
        """Calcular checksum criptográfico"""
        if algorithm == "sha256":
            return hashlib.sha256(data).digest()
        elif algorithm == "sha512":
            return hashlib.sha512(data).digest()
        elif algorithm == "blake2b":
            return hashlib.blake2b(data).digest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def compute_hmac(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """Calcular HMAC"""
        key = key or self.key
        return hmac.new(key, data, hashlib.sha256).digest()
    
    def verify_integrity(self, data: bytes, expected_checksum: bytes, 
                        algorithm: str = "sha256") -> bool:
        """Verificar integridad de datos"""
        computed = self.compute_checksum(data, algorithm)
        return hmac.compare_digest(computed, expected_checksum)
    
    def sign_data(self, data: bytes) -> bytes:
        """Firmar datos con HMAC"""
        return self.compute_hmac(data)
    
    def verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Verificar firma de datos"""
        expected = self.compute_hmac(data)
        return hmac.compare_digest(expected, signature)

class MerkleTree:
    """Árbol Merkle optimizado para verificación de integridad"""
    
    def __init__(self, data_chunks: List[bytes], hash_algorithm: str = "sha256"):
        self.hash_algorithm = hash_algorithm
        self.leaves = [self._hash_chunk(chunk) for chunk in data_chunks]
        self.tree = self._build_tree()
        self.root = self.tree[0] if self.tree else b''
    
    def _hash_chunk(self, chunk: bytes) -> bytes:
        """Calcular hash de un chunk"""
        if self.hash_algorithm == "sha256":
            return hashlib.sha256(chunk).digest()
        elif self.hash_algorithm == "sha512":
            return hashlib.sha512(chunk).digest()
        elif self.hash_algorithm == "blake2b":
            return hashlib.blake2b(chunk).digest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")
    
    def _build_tree(self) -> List[List[bytes]]:
        """Construir árbol Merkle"""
        if not self.leaves:
            return []
        
        current_level = self.leaves.copy()
        tree = [current_level]
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                combined = left + right
                next_level.append(self._hash_chunk(combined))
            current_level = next_level
            tree.append(current_level)
        
        return tree
    
    def get_proof(self, index: int) -> List[bytes]:
        """Obtener prueba de Merkle para un índice"""
        if not self.tree or index >= len(self.leaves):
            return []
        
        proof = []
        current_index = index
        
        for level in self.tree[:-1]:  # Excluir la raíz
            if current_index % 2 == 0:  # Nodo izquierdo
                sibling_index = current_index + 1
            else:  # Nodo derecho
                sibling_index = current_index - 1
            
            if sibling_index < len(level):
                proof.append(level[sibling_index])
            
            current_index //= 2
        
        return proof
    
    def verify_proof(self, leaf: bytes, proof: List[bytes], index: int) -> bool:
        """Verificar prueba de Merkle"""
        current_hash = self._hash_chunk(leaf)
        
        for sibling in proof:
            if index % 2 == 0:
                combined = current_hash + sibling
            else:
                combined = sibling + current_hash
            current_hash = self._hash_chunk(combined)
            index //= 2
        
        return current_hash == self.root
    
    def get_root(self) -> bytes:
        """Obtener raíz del árbol"""
        return self.root

class SecurityAuditor:
    """Auditor de seguridad para MNEME"""
    
    def __init__(self, log_file: Optional[str] = None, 
                 security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.log_file = log_file
        self.security_level = security_level
        self.verifier = CryptographicVerifier(security_level)
        self.audit_logs: List[AuditLog] = []
        self.violations: List[Dict[str, Any]] = []
        
        # Configurar logging de seguridad
        self.security_logger = logging.getLogger("mneme_security")
        self.security_logger.setLevel(logging.INFO)
        
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.security_logger.addHandler(handler)
    
    def log_event(self, event: AuditEvent, resource_id: str, 
                  user_id: Optional[str] = None, details: Dict[str, Any] = None):
        """Registrar evento de auditoría"""
        details = details or {}
        
        # Crear entrada de auditoría
        audit_entry = AuditLog(
            timestamp=time.time(),
            event=event,
            resource_id=resource_id,
            user_id=user_id,
            details=details
        )
        
        # Firmar entrada si el nivel de seguridad lo requiere
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]:
            entry_data = json.dumps(audit_entry.to_dict(), sort_keys=True).encode()
            audit_entry.signature = self.verifier.sign_data(entry_data)
        
        self.audit_logs.append(audit_entry)
        
        # Log a archivo si está configurado
        if self.log_file:
            self.security_logger.info(
                f"Event: {event.value}, Resource: {resource_id}, "
                f"User: {user_id}, Details: {details}"
            )
    
    def verify_resource_integrity(self, resource_id: str, 
                                 data: bytes, 
                                 expected_checksum: bytes) -> bool:
        """Verificar integridad de un recurso"""
        is_valid = self.verifier.verify_integrity(data, expected_checksum)
        
        self.log_event(
            AuditEvent.VERIFY,
            resource_id,
            details={
                "integrity_check": is_valid,
                "data_size": len(data),
                "checksum_algorithm": "sha256"
            }
        )
        
        if not is_valid:
            self._record_violation("integrity_check_failed", resource_id, {
                "expected_checksum": expected_checksum.hex(),
                "data_size": len(data)
            })
        
        return is_valid
    
    def verify_merkle_proof(self, resource_id: str, 
                           leaf_data: bytes, 
                           proof: List[bytes], 
                           index: int,
                           merkle_root: bytes) -> bool:
        """Verificar prueba de Merkle"""
        # Recrear árbol temporal para verificación
        temp_tree = MerkleTree([leaf_data])
        is_valid = temp_tree.verify_proof(leaf_data, proof, index)
        
        # Verificar que la raíz coincida
        if is_valid:
            is_valid = temp_tree.get_root() == merkle_root
        
        self.log_event(
            AuditEvent.VERIFY,
            resource_id,
            details={
                "merkle_verification": is_valid,
                "proof_length": len(proof),
                "index": index
            }
        )
        
        if not is_valid:
            self._record_violation("merkle_verification_failed", resource_id, {
                "proof_length": len(proof),
                "index": index,
                "expected_root": merkle_root.hex()
            })
        
        return is_valid
    
    def _record_violation(self, violation_type: str, resource_id: str, details: Dict[str, Any]):
        """Registrar violación de seguridad"""
        violation = {
            "timestamp": time.time(),
            "type": violation_type,
            "resource_id": resource_id,
            "details": details
        }
        
        self.violations.append(violation)
        
        # Log crítico
        self.security_logger.critical(
            f"SECURITY VIOLATION: {violation_type} for resource {resource_id}"
        )
        
        # Log evento de violación
        self.log_event(
            AuditEvent.SECURITY_VIOLATION,
            resource_id,
            details=violation
        )
    
    def get_security_report(self) -> Dict[str, Any]:
        """Obtener reporte de seguridad"""
        total_events = len(self.audit_logs)
        total_violations = len(self.violations)
        
        # Estadísticas por tipo de evento
        event_counts = {}
        for log in self.audit_logs:
            event_type = log.event.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Estadísticas por tipo de violación
        violation_counts = {}
        for violation in self.violations:
            violation_type = violation["type"]
            violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
        
        return {
            "total_events": total_events,
            "total_violations": total_violations,
            "security_level": self.security_level.value,
            "event_counts": event_counts,
            "violation_counts": violation_counts,
            "violation_rate": total_violations / max(1, total_events),
            "recent_violations": self.violations[-10:] if self.violations else []
        }
    
    def export_audit_logs(self, filepath: str) -> bool:
        """Exportar logs de auditoría"""
        try:
            logs_data = [log.to_dict() for log in self.audit_logs]
            
            with open(filepath, 'w') as f:
                json.dump(logs_data, f, indent=2)
            
            self.log_event(
                AuditEvent.CREATE,
                "audit_export",
                details={"filepath": filepath, "log_count": len(logs_data)}
            )
            
            return True
        except Exception as e:
            self.security_logger.error(f"Failed to export audit logs: {e}")
            return False

class SecureDescriptor:
    """Descriptor seguro con verificación criptográfica"""
    
    def __init__(self, data: bytes, 
                 security_level: SecurityLevel = SecurityLevel.STANDARD,
                 auditor: Optional[SecurityAuditor] = None):
        self.data = data
        self.security_level = security_level
        self.auditor = auditor
        self.verifier = CryptographicVerifier(security_level)
        
        # Calcular verificaciones de seguridad
        self.checksum = self.verifier.compute_checksum(data)
        self.signature = self.verifier.sign_data(data)
        
        # Crear árbol Merkle si el nivel de seguridad lo requiere
        self.merkle_root = None
        if security_level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]:
            merkle_tree = MerkleTree([data])
            self.merkle_root = merkle_tree.get_root()
        
        # Registrar creación
        if auditor:
            auditor.log_event(
                AuditEvent.CREATE,
                f"secure_descriptor_{id(self)}",
                details={
                    "data_size": len(data),
                    "security_level": security_level.value,
                    "has_merkle": self.merkle_root is not None
                }
            )
    
    def verify_integrity(self) -> bool:
        """Verificar integridad del descriptor"""
        is_valid = self.verifier.verify_integrity(self.data, self.checksum)
        
        if self.auditor:
            self.auditor.log_event(
                AuditEvent.VERIFY,
                f"secure_descriptor_{id(self)}",
                details={"integrity_check": is_valid}
            )
        
        return is_valid
    
    def verify_signature(self) -> bool:
        """Verificar firma del descriptor"""
        is_valid = self.verifier.verify_signature(self.data, self.signature)
        
        if self.auditor:
            self.auditor.log_event(
                AuditEvent.VERIFY,
                f"secure_descriptor_{id(self)}",
                details={"signature_check": is_valid}
            )
        
        return is_valid
    
    def get_merkle_proof(self, chunk_index: int = 0) -> Optional[List[bytes]]:
        """Obtener prueba de Merkle para un chunk"""
        if not self.merkle_root:
            return None
        
        merkle_tree = MerkleTree([self.data])
        return merkle_tree.get_proof(chunk_index)
    
    def verify_merkle_proof(self, proof: List[bytes], index: int) -> bool:
        """Verificar prueba de Merkle"""
        if not self.merkle_root:
            return False
        
        merkle_tree = MerkleTree([self.data])
        return merkle_tree.verify_proof(self.data, proof, index)

class SecurityManager:
    """Gestor de seguridad centralizado para MNEME"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD,
                 audit_log_file: Optional[str] = None):
        self.security_level = security_level
        self.auditor = SecurityAuditor(audit_log_file, security_level)
        self.verifier = CryptographicVerifier(security_level)
        
        # Configuración de políticas de seguridad
        self.policies = {
            "max_failed_attempts": 5,
            "lockout_duration": 300,  # 5 minutos
            "require_signatures": security_level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM],
            "require_merkle": security_level == SecurityLevel.MAXIMUM
        }
        
        # Estado de seguridad
        self.failed_attempts = {}
        self.locked_resources = set()
        self.active_sessions = {}
        
        logger.info(f"Security Manager initialized with level: {security_level.name}")
    
    def create_secure_descriptor(self, data: bytes, resource_id: str) -> SecureDescriptor:
        """Crear descriptor seguro"""
        descriptor = SecureDescriptor(data, self.security_level, self.auditor)
        
        self.auditor.log_event(
            AuditEvent.CREATE,
            resource_id,
            details={
                "descriptor_id": id(descriptor),
                "data_size": len(data),
                "security_level": self.security_level.value
            }
        )
        
        return descriptor
    
    def verify_resource_access(self, resource_id: str, user_id: Optional[str] = None) -> bool:
        """Verificar acceso a recurso"""
        # Verificar si el recurso está bloqueado
        if resource_id in self.locked_resources:
            self.auditor.log_event(
                AuditEvent.SECURITY_VIOLATION,
                resource_id,
                user_id,
                {"reason": "resource_locked"}
            )
            return False
        
        # Verificar intentos fallidos
        if resource_id in self.failed_attempts:
            attempts, last_attempt = self.failed_attempts[resource_id]
            if attempts >= self.policies["max_failed_attempts"]:
                if time.time() - last_attempt < self.policies["lockout_duration"]:
                    self.locked_resources.add(resource_id)
                    return False
                else:
                    # Resetear intentos fallidos
                    del self.failed_attempts[resource_id]
                    self.locked_resources.discard(resource_id)
        
        return True
    
    def record_failed_access(self, resource_id: str, user_id: Optional[str] = None):
        """Registrar acceso fallido"""
        if resource_id not in self.failed_attempts:
            self.failed_attempts[resource_id] = [0, time.time()]
        
        self.failed_attempts[resource_id][0] += 1
        self.failed_attempts[resource_id][1] = time.time()
        
        self.auditor.log_event(
            AuditEvent.SECURITY_VIOLATION,
            resource_id,
            user_id,
            {"reason": "failed_access", "attempts": self.failed_attempts[resource_id][0]}
        )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Obtener estado de seguridad"""
        return {
            "security_level": self.security_level.value,
            "policies": self.policies,
            "locked_resources": list(self.locked_resources),
            "failed_attempts": len(self.failed_attempts),
            "active_sessions": len(self.active_sessions),
            "audit_report": self.auditor.get_security_report()
        }
    
    def cleanup_expired_locks(self):
        """Limpiar bloqueos expirados"""
        current_time = time.time()
        expired_resources = []
        
        for resource_id in self.locked_resources:
            if resource_id in self.failed_attempts:
                _, last_attempt = self.failed_attempts[resource_id]
                if current_time - last_attempt > self.policies["lockout_duration"]:
                    expired_resources.append(resource_id)
        
        for resource_id in expired_resources:
            self.locked_resources.discard(resource_id)
            if resource_id in self.failed_attempts:
                del self.failed_attempts[resource_id]
        
        if expired_resources:
            logger.info(f"Cleaned up {len(expired_resources)} expired locks")
