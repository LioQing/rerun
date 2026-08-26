use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use ewebsock::{WsEvent, WsMessage};
use macaw::{Quat, Vec3};
use re_byte_size::SizeBytes;
use re_log_types::EntityPath;
use re_mutex::Mutex;
use re_viewer_context::{InteractConnectionStatus, InteractOptions};
use url::Url;

const MAX_BACKOFF: Duration = Duration::from_secs(30);
const MAX_SEND_RATE: Duration = Duration::from_millis(100);

static STATE: std::sync::LazyLock<Mutex<InteractState>> =
    std::sync::LazyLock::new(|| Mutex::new(InteractState::default()));

pub fn state() -> re_mutex::MutexGuard<'static, InteractState> {
    STATE.lock()
}

/// User interaction thread message.
///
/// For controlling the thread.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize, SizeBytes)]
pub enum InteractThreadMessage {
    /// Send user interaction data to the thread.
    Data(InteractData),
    /// Stop the thread.
    Stop,
}

/// User interaction data recorded.
///
/// This data is serialized and sent through websocket.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize, SizeBytes)]
pub struct InteractData {
    pub path: EntityPath,
    pub eye_rotation: Quat,
    pub eye_position: Vec3,
    pub eye_fov_y: Option<f32>,
}

/// State of the interaction feature.
///
/// If this is `None`, client has not connected to a websocket for
/// interaction streaming.
#[derive(Default)]
pub struct InteractState(Arc<Mutex<Option<InteractWsHandle>>>);

impl InteractState {
    pub fn update(&self, options: &InteractOptions) -> Result<(), Box<dyn std::error::Error>> {
        self.ensure_connected(options)?;
        self.update_connection_status(options);

        Ok(())
    }

    pub fn send(&self, data: InteractData) {
        let guard = self.0.lock();
        if let Some(handle) = guard.as_ref() {
            handle.send(data);
        }
    }

    fn update_connection_status(&self, options: &InteractOptions) {
        let guard = self.0.lock();
        if let Some(handle) = guard.as_ref() {
            if handle
                .is_connected
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                *options.connection_status.lock() = InteractConnectionStatus::Connected;
            } else {
                *options.connection_status.lock() = InteractConnectionStatus::Connecting;
            }
        }
    }

    fn ensure_connected(
        &self,
        options: &InteractOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let connection_status = *options.connection_status.lock();
        if connection_status != InteractConnectionStatus::Disconnected && !self.is_connected() {
            let address = Url::parse(&options.address)?;
            let handle = InteractWsHandle::new(address)?;

            let mut guard = self.0.lock();
            *guard = Some(handle);
        } else if connection_status == InteractConnectionStatus::Disconnected && self.is_connected()
        {
            let mut guard = self.0.lock();
            if let Some(handle) = guard.take() {
                handle.disconnect();
            }
        }

        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.0
            .lock()
            .as_ref()
            .map(|handle| {
                handle
                    .is_connected
                    .load(std::sync::atomic::Ordering::Relaxed)
            })
            .unwrap_or(false)
    }
}

/// The handle to the websocket connection for interaction streaming.
pub struct InteractWsHandle {
    pub handle: Option<std::thread::JoinHandle<()>>,
    pub tx: re_quota_channel::Sender<InteractThreadMessage>,
    pub is_connected: Arc<AtomicBool>,
}

impl InteractWsHandle {
    pub fn new(address: Url) -> Result<Self, std::io::Error> {
        let (tx, rx) =
            re_quota_channel::channel::<InteractThreadMessage>("interact-thread-message", 512);
        let is_connected = Arc::new(AtomicBool::new(false));

        let handle = std::thread::Builder::new()
            .name("InteractWsThread".into())
            .spawn({
                let is_connected = Arc::clone(&is_connected);
                move || {
                    run_interact_ws_thread(&rx, &address, &is_connected);
                }
            })?;

        Ok(Self {
            handle: Some(handle),
            tx,
            is_connected,
        })
    }

    pub fn send(&self, data: InteractData) {
        if let Err(err) = self.tx.send(InteractThreadMessage::Data(data)) {
            re_log::warn_once!("Failed to send interaction data: {err}");
        }
    }

    pub fn disconnect(self) {
        if let Some(handle) = self.handle {
            if let Err(err) = self.tx.send(InteractThreadMessage::Stop) {
                re_log::warn_once!("Failed to send stop message to interaction thread: {err}");
            }

            handle.join().ok();
        }
    }
}

/// The websocket thread for interaction streaming.
fn run_interact_ws_thread(
    rx: &re_quota_channel::Receiver<InteractThreadMessage>,
    address: &Url,
    is_connected: &AtomicBool,
) {
    is_connected.store(false, std::sync::atomic::Ordering::Relaxed);
    let mut backoff = Duration::from_secs(1);
    let mut last_data = None;
    let mut last_sent = Instant::now();

    // Connection loop: try to connect, and if we fail, wait with exponential backoff and try again.
    loop {
        // If a Stop has already been signalled, exit immediately.
        match rx.try_recv() {
            Ok(InteractThreadMessage::Stop) | Err(re_quota_channel::TryRecvError::Disconnected) => {
                return;
            }
            Ok(InteractThreadMessage::Data(_)) | Err(re_quota_channel::TryRecvError::Empty) => {
                // There is data queued before we've connected; we'll connect and handle it.
            }
        }

        re_log::info!("Connecting to interact websocket at {address}...");
        match ewebsock::connect(address.clone(), ewebsock::Options::default()) {
            Ok((mut sender, receiver)) => {
                // Successful connection
                is_connected.store(true, std::sync::atomic::Ordering::Relaxed);
                backoff = Duration::from_secs(1);

                // Connected loop: wait for either channel messages or websocket state changes.
                loop {
                    match rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(InteractThreadMessage::Data(data)) => {
                            // Don't send data too frequently.
                            if last_sent.elapsed() < MAX_SEND_RATE {
                                continue;
                            }

                            // Don't send the same data twice in a row.
                            if last_data.as_ref() == Some(&data) {
                                continue;
                            }

                            last_sent = Instant::now();
                            last_data = Some(data.clone());

                            match serde_json::to_string(&data) {
                                Ok(text) => {
                                    sender.send(WsMessage::Text(text));
                                }
                                Err(err) => {
                                    re_log::warn_once!(
                                        "Failed to serialize interaction data: {err}"
                                    );
                                }
                            }
                        }
                        Err(re_quota_channel::RecvTimeoutError::Timeout) => {
                            // Periodically check websocket events.
                            if let Some(event) = receiver.try_recv() {
                                match event {
                                    WsEvent::Closed => {
                                        re_log::debug!("Interact WS closed by server");
                                        break; // reconnect
                                    }
                                    WsEvent::Error(err) => {
                                        re_log::warn_once!("Interact WS error: {err}");
                                        break; // reconnect
                                    }
                                    _ => {}
                                }
                            }
                            // Continue waiting for channel messages.
                        }
                        Ok(InteractThreadMessage::Stop)
                        | Err(re_quota_channel::RecvTimeoutError::Disconnected) => {
                            drop(sender);
                            return; // disconnect
                        }
                    }
                } // end connected loop
            }
            Err(err) => {
                re_log::warn_once!("Failed to connect to interact websocket at {address}: {err}");
            }
        }

        // If we get here, we either failed to connect or the connection dropped.
        // Wait with exponential backoff, but still check for Stop or Disconnect.
        is_connected.store(false, std::sync::atomic::Ordering::Relaxed);
        let start = Instant::now();
        while start.elapsed() < backoff {
            match rx.try_recv() {
                Ok(InteractThreadMessage::Stop)
                | Err(re_quota_channel::TryRecvError::Disconnected) => return,
                Ok(InteractThreadMessage::Data(_)) | Err(re_quota_channel::TryRecvError::Empty) => {
                    // drop / ignore data while disconnected
                }
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        backoff = std::cmp::min(backoff * 2, MAX_BACKOFF);
    }
}
