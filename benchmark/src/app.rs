/// Inspired by https://github.com/hatoo/oha/blob/bb989ea3cd77727e7743e7daa60a19894bb5e901/src/monitor.rs
use crate::embedder::{Message, Embed};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use embedding_server_client::ClientError;
use tokio::sync::mpsc;
use tui::backend::Backend;
use tui::layout::{Alignment, Constraint, Direction, Layout};
use tui::style::{Color, Modifier, Style};
use tui::text::{Span, Spans};
use tui::widgets::{
    Axis, BarChart, Block, Borders, Chart, Dataset, Gauge, GraphType, Paragraph, Tabs,
};
use tui::{symbols, Frame};

/// TUI powered App
pub(crate) struct App {
    pub(crate) running: bool,
    completed_runs: Vec<usize>,
    completed_batch: usize,
    current_batch: usize,
    current_tab: usize,
    touched_tab: bool,
    zoom: bool,
    is_error: bool,
    data: Data,
    tokenizer_name: String,
    sequence_length: u32,
    n_run: usize,
    batch_size: Vec<u32>,
    receiver: mpsc::Receiver<Result<Message, ClientError>>,
}

impl App {
    pub(crate) fn new(
        receiver: mpsc::Receiver<Result<Message, ClientError>>,
        tokenizer_name: String,
        sequence_length: u32,
        n_run: usize,
        batch_size: Vec<u32>,
    ) -> Self {
        let data = Data::new(n_run, batch_size.len());
        let current_tab = 0;

        let completed_runs: Vec<usize> = (0..batch_size.len()).map(|_| 0).collect();
        let completed_batch = 0;
        let current_batch = 0;
        let is_error = false;

        Self {
            running: true,
            completed_runs,
            completed_batch,
            current_batch,
            current_tab,
            touched_tab: false,
            zoom: false,
            is_error,
            data,
            tokenizer_name,
            sequence_length,
            n_run,
            batch_size,
            receiver,
        }
    }

    /// Handle crossterm key events
    pub(crate) fn handle_key_event(&mut self, key_event: KeyEvent) {
        match key_event {
            // Increase and wrap tab
            KeyEvent {
                code: KeyCode::Right,
                ..
            }
            | KeyEvent {
                code: KeyCode::Tab, ..
            } => {
                self.touched_tab = true;
                self.current_tab = (self.current_tab + 1) % self.batch_size.len();
            }
            // Decrease and wrap tab
            KeyEvent {
                code: KeyCode::Left,
                ..
            } => {
                self.touched_tab = true;
                if self.current_tab > 0 {
                    self.current_tab -= 1;
                } else {
                    self.current_tab = self.batch_size.len() - 1;
                }
            }
            // Zoom on throughput/latency fig
            KeyEvent {
                code: KeyCode::Char('+'),
                ..
            } => {
                self.zoom = true;
            }
            // Unzoom on throughput/latency fig
            KeyEvent {
                code: KeyCode::Char('-'),
                ..
            } => {
                self.zoom = false;
            }
            // Quit
            KeyEvent {
                code: KeyCode::Char('q'),
                ..
            }
            | KeyEvent {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.running = false;
            }
            _ => (),
        }
    }

    /// Get all pending messages from generation task
    pub(crate) fn tick(&mut self) {
        while let Ok(message) = self.receiver.try_recv() {
            match message {
                Ok(message) => match message {
                    Message::Embed(step) => self.data.push_embed(step, self.current_batch),
                    Message::EndRun => {
                        self.completed_runs[self.current_batch] += 1;
                    }
                    Message::EndBatch => {
                        self.data.end_batch(self.current_batch);
                        self.completed_batch += 1;

                        if self.current_batch < self.batch_size.len() - 1 {
                            // Only go to next tab if the user never touched the tab keys
                            if !self.touched_tab {
                                self.current_tab += 1;
                            }

                            self.current_batch += 1;
                        }
                    }
                    Message::Warmup => {}
                },
                Err(_) => self.is_error = true,
            }
        }
    }

    /// Render frame
    pub fn render<B: Backend>(&mut self, f: &mut Frame<'_, B>) {
        let batch_progress =
            (self.completed_batch as f64 / self.batch_size.len() as f64).clamp(0.0, 1.0);
        let run_progress =
            (self.completed_runs[self.current_batch] as f64 / self.n_run as f64).clamp(0.0, 1.0);

        // Vertical layout
        let row5 = Layout::default()
            .direction(Direction::Vertical)
            .constraints(
                [
                    Constraint::Length(1),
                    Constraint::Length(3),
                    Constraint::Length(3),
                    Constraint::Length(13),
                    Constraint::Min(10),
                ]
                .as_ref(),
            )
            .split(f.size());

        // Top row horizontal layout
        let top = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
            .split(row5[2]);

        // Mid row horizontal layout
        let mid = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                [
                    Constraint::Percentage(50),
                    Constraint::Percentage(50),
                ]
                .as_ref(),
            )
            .split(row5[3]);

        // Left mid row vertical layout
        let embed_text = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(8), Constraint::Length(5)].as_ref())
            .split(mid[0]);

        // Bottom row horizontal layout
        let bottom = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(100)].as_ref())
            .split(row5[4]);

        // Title
        let title = Block::default()
            .borders(Borders::NONE)
            .title(format!(
                "Model: {} | Sequence Length: {}",
                self.tokenizer_name, self.sequence_length
            ))
            .style(
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .fg(Color::White),
            );
        f.render_widget(title, row5[0]);

        // Helper
        let helper = Block::default()
            .borders(Borders::NONE)
            .title("<- | tab | ->: change batch tab | q / CTRL + c: quit | +/-: zoom")
            .title_alignment(Alignment::Right)
            .style(Style::default().fg(Color::White));
        f.render_widget(helper, row5[0]);

        // Batch tabs
        let titles = self
            .batch_size
            .iter()
            .map(|b| {
                Spans::from(vec![Span::styled(
                    format!("Batch: {b}"),
                    Style::default().fg(Color::White),
                )])
            })
            .collect();
        let tabs = Tabs::new(titles)
            .block(Block::default().borders(Borders::ALL).title("Tabs"))
            .select(self.current_tab)
            .style(Style::default().fg(Color::LightCyan))
            .highlight_style(
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .bg(Color::Black),
            );
        f.render_widget(tabs, row5[1]);

        // Total progress bar
        let color = if self.is_error {
            Color::Red
        } else {
            Color::LightGreen
        };
        let batch_gauge = progress_gauge(
            "Total Progress",
            format!("{} / {}", self.completed_batch, self.batch_size.len()),
            batch_progress,
            color,
        );
        f.render_widget(batch_gauge, top[0]);

        // Batch progress Bar
        let color = if self.is_error {
            Color::Red
        } else {
            Color::LightBlue
        };
        let run_gauge = progress_gauge(
            "Batch Progress",
            format!(
                "{} / {}",
                self.completed_runs[self.current_batch], self.n_run
            ),
            run_progress,
            color,
        );
        f.render_widget(run_gauge, top[1]);

        // Embed text infos
        let embed_latency_block = latency_paragraph(
            &mut self.data.embed_latencies[self.current_tab],
            "Embed",
        );
        let embed_throughput_block =
            throughput_paragraph(&self.data.embed_throughputs[self.current_tab], "Embed");

        f.render_widget(embed_latency_block, embed_text[0]);
        f.render_widget(embed_throughput_block, embed_text[1]);

        // Embed latency histogram
        let histo_width = 7;
        let bins = if mid[1].width < 2 {
            0
        } else {
            (mid[1].width as usize - 2) / (histo_width + 1)
        }
        .max(2);

        let histo_data =
            latency_histogram_data(&self.data.embed_latencies[self.current_tab], bins);
        let histo_data_str: Vec<(&str, u64)> =
            histo_data.iter().map(|(l, v)| (l.as_str(), *v)).collect();
        let embed_histogram =
            latency_histogram(&histo_data_str, "Embed").bar_width(histo_width as u16);
        f.render_widget(embed_histogram, mid[1]);

        // Embed latency/throughput chart
        let embed_latency_throughput_chart = latency_throughput_chart(
            &self.data.embed_batch_latency_throughput,
            &self.batch_size,
            self.zoom,
            "Embed",
        );
        f.render_widget(embed_latency_throughput_chart, bottom[0]);
    }
}

/// App internal data struct
struct Data {
    embed_latencies: Vec<Vec<f64>>,
    embed_throughputs: Vec<Vec<f64>>,
    embed_batch_latency_throughput: Vec<(f64, f64)>,
}

impl Data {
    fn new(n_run: usize, n_batch: usize) -> Self {
        let embed_latencies: Vec<Vec<f64>> =
            (0..n_batch).map(|_| Vec::with_capacity(n_run)).collect();
        let embed_throughputs: Vec<Vec<f64>> = embed_latencies.clone();

        let embed_batch_latency_throughput: Vec<(f64, f64)> = Vec::with_capacity(n_batch);

        Self {
            embed_latencies,
            embed_throughputs,
            embed_batch_latency_throughput,
        }
    }

    fn push_embed(&mut self, embed: Embed, batch_idx: usize) {
        let latency = embed.latency.as_millis() as f64;
        self.embed_latencies[batch_idx].push(latency);
        self.embed_throughputs[batch_idx].push(embed.throughput);
    }

    fn end_batch(&mut self, batch_idx: usize) {
        self.embed_batch_latency_throughput.push((
            self.embed_latencies[batch_idx].iter().sum::<f64>()
                / self.embed_latencies[batch_idx].len() as f64,
            self.embed_throughputs[batch_idx].iter().sum::<f64>()
                / self.embed_throughputs[batch_idx].len() as f64,
        ));
    }
}

/// Progress bar
fn progress_gauge(title: &str, label: String, progress: f64, color: Color) -> Gauge {
    Gauge::default()
        .block(Block::default().title(title).borders(Borders::ALL))
        .gauge_style(Style::default().fg(color))
        .label(Span::raw(label))
        .ratio(progress)
}

/// Throughput paragraph
fn throughput_paragraph<'a>(throughput: &Vec<f64>, name: &'static str) -> Paragraph<'a> {
    // Throughput average/high/low texts
    let throughput_texts = statis_spans(throughput, "tokens/secs");

    // Throughput block
    Paragraph::new(throughput_texts).block(
        Block::default()
            .title(Span::raw(format!("{name} Throughput")))
            .borders(Borders::ALL),
    )
}

/// Latency paragraph
fn latency_paragraph<'a>(latency: &mut Vec<f64>, name: &'static str) -> Paragraph<'a> {
    // Latency average/high/low texts
    let mut latency_texts = statis_spans(latency, "ms");

    // Sort latency for percentiles
    float_ord::sort(latency);
    let latency_percentiles = crate::utils::percentiles(latency, &[50, 90, 99]);

    // Latency p50/p90/p99 texts
    let colors = vec![Color::LightGreen, Color::LightYellow, Color::LightRed];
    for (i, (name, value)) in latency_percentiles.iter().enumerate() {
        let span = Spans::from(vec![Span::styled(
            format!("{name}:     {value:.2} ms"),
            Style::default().fg(colors[i]),
        )]);
        latency_texts.push(span);
    }

    Paragraph::new(latency_texts).block(
        Block::default()
            .title(Span::raw(format!("{name} Latency")))
            .borders(Borders::ALL),
    )
}

/// Average/High/Low spans
fn statis_spans<'a>(data: &Vec<f64>, unit: &'static str) -> Vec<Spans<'a>> {
    vec![
        Spans::from(vec![Span::styled(
            format!(
                "Average: {:.2} {unit}",
                data.iter().sum::<f64>() / data.len() as f64
            ),
            Style::default().fg(Color::LightBlue),
        )]),
        Spans::from(vec![Span::styled(
            format!(
                "Lowest:  {:.2} {unit}",
                data.iter()
                    .min_by(|a, b| a.total_cmp(b))
                    .unwrap_or(&std::f64::NAN)
            ),
            Style::default().fg(Color::Reset),
        )]),
        Spans::from(vec![Span::styled(
            format!(
                "Highest: {:.2} {unit}",
                data.iter()
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap_or(&std::f64::NAN)
            ),
            Style::default().fg(Color::Reset),
        )]),
    ]
}

/// Latency histogram data
fn latency_histogram_data(latency: &[f64], bins: usize) -> Vec<(String, u64)> {
    let histo_data: Vec<(String, u64)> = {
        let histo = crate::utils::histogram(latency, bins);
        histo
            .into_iter()
            .map(|(label, v)| (format!("{label:.2}"), v as u64))
            .collect()
    };

    histo_data
}

/// Latency Histogram
fn latency_histogram<'a>(
    histo_data_str: &'a Vec<(&'a str, u64)>,
    name: &'static str,
) -> BarChart<'a> {
    BarChart::default()
        .block(
            Block::default()
                .title(format!("{name} latency histogram"))
                .style(Style::default().fg(Color::LightYellow).bg(Color::Reset))
                .borders(Borders::ALL),
        )
        .data(histo_data_str.as_slice())
}

/// Latency/Throughput chart
fn latency_throughput_chart<'a>(
    latency_throughput: &'a Vec<(f64, f64)>,
    batch_sizes: &'a [u32],
    zoom: bool,
    name: &'static str,
) -> Chart<'a> {
    let latency_iter = latency_throughput.iter().map(|(l, _)| l);
    let throughput_iter = latency_throughput.iter().map(|(_, t)| t);

    // Get extreme values
    let min_latency: f64 = *latency_iter
        .clone()
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(&std::f64::NAN);
    let max_latency: f64 = *latency_iter
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(&std::f64::NAN);
    let min_throughput: f64 = *throughput_iter
        .clone()
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(&std::f64::NAN);
    let max_throughput: f64 = *throughput_iter
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(&std::f64::NAN);

    // Char min max values
    let min_x = if zoom {
        ((min_latency - 0.05 * min_latency) / 100.0).floor() * 100.0
    } else {
        0.0
    };
    let max_x = ((max_latency + 0.05 * max_latency) / 100.0).ceil() * 100.0;
    let step_x = (max_x - min_x) / 4.0;

    // Chart min max values
    let min_y = if zoom {
        ((min_throughput - 0.05 * min_throughput) / 100.0).floor() * 100.0
    } else {
        0.0
    };
    let max_y = ((max_throughput + 0.05 * max_throughput) / 100.0).ceil() * 100.0;
    let step_y = (max_y - min_y) / 4.0;

    // Labels
    let mut x_labels = vec![Span::styled(
        format!("{min_x:.2}"),
        Style::default()
            .add_modifier(Modifier::BOLD)
            .fg(Color::Gray)
            .bg(Color::Reset),
    )];
    for i in 0..3 {
        x_labels.push(Span::styled(
            format!("{:.2}", min_x + ((i + 1) as f64 * step_x)),
            Style::default().fg(Color::Gray).bg(Color::Reset),
        ));
    }
    x_labels.push(Span::styled(
        format!("{max_x:.2}"),
        Style::default()
            .add_modifier(Modifier::BOLD)
            .fg(Color::Gray)
            .bg(Color::Reset),
    ));

    // Labels
    let mut y_labels = vec![Span::styled(
        format!("{min_y:.2}"),
        Style::default()
            .add_modifier(Modifier::BOLD)
            .fg(Color::Gray)
            .bg(Color::Reset),
    )];
    for i in 0..3 {
        y_labels.push(Span::styled(
            format!("{:.2}", min_y + ((i + 1) as f64 * step_y)),
            Style::default().fg(Color::Gray).bg(Color::Reset),
        ));
    }
    y_labels.push(Span::styled(
        format!("{max_y:.2}"),
        Style::default()
            .add_modifier(Modifier::BOLD)
            .fg(Color::Gray)
            .bg(Color::Reset),
    ));

    // Chart dataset
    let colors = color_vec();
    let datasets: Vec<Dataset> = (0..latency_throughput.len())
        .map(|i| {
            let color_idx = i % colors.len();

            Dataset::default()
                .name(batch_sizes[i].to_string())
                .marker(symbols::Marker::Block)
                .style(Style::default().fg(colors[color_idx]))
                .graph_type(GraphType::Scatter)
                .data(&latency_throughput[i..(i + 1)])
        })
        .collect();

    // Chart
    Chart::new(datasets)
        .style(Style::default().fg(Color::Cyan).bg(Color::Reset))
        .block(
            Block::default()
                .title(Span::styled(
                    format!("{name} throughput over latency"),
                    Style::default().fg(Color::Gray).bg(Color::Reset),
                ))
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("ms")
                .style(Style::default().fg(Color::Gray).bg(Color::Reset))
                .labels(x_labels)
                .bounds([min_x, max_x]),
        )
        .y_axis(
            Axis::default()
                .title("tokens/secs")
                .style(Style::default().fg(Color::Gray).bg(Color::Reset))
                .labels(y_labels)
                .bounds([min_y, max_y]),
        )
}

// Colors for latency/throughput chart
fn color_vec() -> Vec<Color> {
    vec![
        Color::Red,
        Color::Green,
        Color::Yellow,
        Color::Blue,
        Color::Magenta,
        Color::Cyan,
        Color::Gray,
        Color::DarkGray,
        Color::LightRed,
        Color::LightGreen,
        Color::LightYellow,
        Color::LightBlue,
        Color::LightMagenta,
        Color::LightCyan,
    ]
}
