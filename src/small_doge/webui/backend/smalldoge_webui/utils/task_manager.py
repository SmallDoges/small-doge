# Copyright 2025 The SmallDoge Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Task Manager for SmallDoge WebUI
Handles inference task cancellation and tracking
"""

import asyncio
import threading
import time
import uuid
from typing import Dict, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import logging

log = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class InferenceTask:
    """Represents an inference task"""
    task_id: str
    user_id: Optional[str]  # For future use, currently None for open access
    model_id: str
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    cancellation_event: Optional[asyncio.Event] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class TaskManager:
    """
    Manages inference tasks with cancellation support
    Provides thread-safe task tracking and cancellation
    """

    def __init__(self):
        self._tasks: Dict[str, InferenceTask] = {}
        self._lock = threading.RLock()
        self._cleanup_interval = 300  # 5 minutes
        self._max_task_age = 3600  # 1 hour
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Start the task manager"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        log.info("Task manager started")

    async def stop(self):
        """Stop the task manager"""
        self._shutdown_event.set()
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all running tasks
        await self.cancel_all_tasks()
        log.info("Task manager stopped")

    def create_task(
        self,
        model_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """
        Create a new inference task
        
        Args:
            model_id: Model identifier
            user_id: User identifier (optional, for open access)
        
        Returns:
            str: Task ID
        """
        task_id = str(uuid.uuid4())
        
        with self._lock:
            cancellation_event = asyncio.Event()
            task = InferenceTask(
                task_id=task_id,
                user_id=user_id,
                model_id=model_id,
                status=TaskStatus.PENDING,
                created_at=time.time(),
                cancellation_event=cancellation_event
            )
            self._tasks[task_id] = task
            
        log.info(f"Created inference task {task_id} for model {model_id}")
        return task_id

    def start_task(self, task_id: str) -> bool:
        """
        Mark task as started
        
        Args:
            task_id: Task identifier
        
        Returns:
            bool: True if task was started, False if not found or already started
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.PENDING:
                return False
            
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            
        log.info(f"Started inference task {task_id}")
        return True

    def complete_task(
        self,
        task_id: str,
        result: Optional[Any] = None,
        error: Optional[str] = None
    ) -> bool:
        """
        Mark task as completed
        
        Args:
            task_id: Task identifier
            result: Task result (optional)
            error: Error message if task failed (optional)
        
        Returns:
            bool: True if task was completed, False if not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            
            task.status = TaskStatus.FAILED if error else TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result
            task.error = error
            
        status_str = "failed" if error else "completed"
        log.info(f"Task {task_id} {status_str}")
        return True

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running inference task
        
        Args:
            task_id: Task identifier
        
        Returns:
            bool: True if task was cancelled, False if not found or not cancellable
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                return False
            
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            
            # Signal cancellation
            if task.cancellation_event:
                task.cancellation_event.set()
        
        log.info(f"Cancelled inference task {task_id}")
        return True

    async def cancel_all_tasks(self) -> int:
        """
        Cancel all running tasks
        
        Returns:
            int: Number of tasks cancelled
        """
        cancelled_count = 0
        
        with self._lock:
            task_ids = list(self._tasks.keys())
        
        for task_id in task_ids:
            if await self.cancel_task(task_id):
                cancelled_count += 1
        
        log.info(f"Cancelled {cancelled_count} tasks")
        return cancelled_count

    def get_task(self, task_id: str) -> Optional[InferenceTask]:
        """
        Get task by ID
        
        Args:
            task_id: Task identifier
        
        Returns:
            Optional[InferenceTask]: Task if found, None otherwise
        """
        with self._lock:
            return self._tasks.get(task_id)

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get task status
        
        Args:
            task_id: Task identifier
        
        Returns:
            Optional[TaskStatus]: Task status if found, None otherwise
        """
        task = self.get_task(task_id)
        return task.status if task else None

    def is_task_cancelled(self, task_id: str) -> bool:
        """
        Check if task is cancelled
        
        Args:
            task_id: Task identifier
        
        Returns:
            bool: True if task is cancelled, False otherwise
        """
        return self.get_task_status(task_id) == TaskStatus.CANCELLED

    def get_cancellation_event(self, task_id: str) -> Optional[asyncio.Event]:
        """
        Get cancellation event for task
        
        Args:
            task_id: Task identifier
        
        Returns:
            Optional[asyncio.Event]: Cancellation event if found, None otherwise
        """
        task = self.get_task(task_id)
        return task.cancellation_event if task else None

    def get_active_tasks(self) -> Dict[str, InferenceTask]:
        """
        Get all active (pending/running) tasks
        
        Returns:
            Dict[str, InferenceTask]: Active tasks
        """
        with self._lock:
            return {
                task_id: task
                for task_id, task in self._tasks.items()
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
            }

    def get_task_stats(self) -> Dict[str, int]:
        """
        Get task statistics
        
        Returns:
            Dict[str, int]: Task counts by status
        """
        stats = {status.value: 0 for status in TaskStatus}
        
        with self._lock:
            for task in self._tasks.values():
                stats[task.status.value] += 1
        
        return stats

    async def _cleanup_loop(self):
        """Background cleanup loop for old tasks"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._cleanup_interval
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                # Perform cleanup
                await self._cleanup_old_tasks()

    async def _cleanup_old_tasks(self):
        """Remove old completed/failed/cancelled tasks"""
        current_time = time.time()
        cleanup_count = 0
        
        with self._lock:
            tasks_to_remove = []
            for task_id, task in self._tasks.items():
                # Only cleanup completed/failed/cancelled tasks
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    task_age = current_time - task.created_at
                    if task_age > self._max_task_age:
                        tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self._tasks[task_id]
                cleanup_count += 1
        
        if cleanup_count > 0:
            log.info(f"Cleaned up {cleanup_count} old tasks")


# Global task manager instance
task_manager = TaskManager() 