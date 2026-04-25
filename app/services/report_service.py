"""
Report generation: PDF and CSV exports.
"""
import csv
import io
from datetime import date
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.diagnosis import Diagnosis, DiagnosisStatus


class ReportService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def _fetch_diagnoses(
        self,
        date_from: Optional[date],
        date_to: Optional[date],
        facility_name: Optional[str],
    ) -> List[Diagnosis]:
        query = (
            select(Diagnosis)
            .options(
                selectinload(Diagnosis.patient),
                selectinload(Diagnosis.result),
                selectinload(Diagnosis.created_by),
            )
            .where(Diagnosis.status == DiagnosisStatus.COMPLETED)
        )
        if date_from:
            query = query.where(Diagnosis.created_at >= date_from)
        if date_to:
            query = query.where(Diagnosis.created_at <= date_to)
        if facility_name:
            query = query.where(Diagnosis.facility_name == facility_name)
        result = await self.db.execute(query.order_by(Diagnosis.created_at.desc()))
        return result.scalars().all()

    # ── CSV ───────────────────────────────────────────────────────────────────

    async def generate_csv(
        self,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        facility_name: Optional[str] = None,
    ) -> bytes:
        diagnoses = await self._fetch_diagnoses(date_from, date_to, facility_name)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Diagnosis ID", "Date", "Patient Code", "Patient Name",
            "Facility", "Severity", "Parasitaemia (%)",
            "Total Parasites", "RBC Count",
            "Ring", "Trophozoite", "Schizont", "Gametocyte",
            "Technician", "Inference Time (ms)",
        ])

        for d in diagnoses:
            r = d.result
            writer.writerow([
                str(d.id),
                d.created_at.date().isoformat(),
                d.patient.patient_code if d.patient else "",
                d.patient.full_name if d.patient else "",
                d.facility_name or "",
                d.severity.value if d.severity else "",
                r.parasitaemia_percent if r else "",
                r.total_parasite_count if r else "",
                r.total_rbc_count if r else "",
                r.ring_count if r else "",
                r.trophozoite_count if r else "",
                r.schizont_count if r else "",
                r.gametocyte_count if r else "",
                d.created_by.full_name if d.created_by else "",
                r.inference_time_ms if r else "",
            ])

        return output.getvalue().encode("utf-8")

    # ── PDF ───────────────────────────────────────────────────────────────────

    async def generate_pdf(
        self,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        facility_name: Optional[str] = None,
    ) -> bytes:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        diagnoses = await self._fetch_diagnoses(date_from, date_to, facility_name)
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(A4))
        styles = getSampleStyleSheet()
        elements = []

        # Title
        title = f"VisionDx Malaria Diagnostic Report"
        subtitle = f"Period: {date_from or 'all'} → {date_to or 'now'}"
        if facility_name:
            subtitle += f" | Facility: {facility_name}"

        elements.append(Paragraph(title, styles["Title"]))
        elements.append(Paragraph(subtitle, styles["Normal"]))
        elements.append(Spacer(1, 0.5 * cm))

        # Summary stats
        total = len(diagnoses)
        positive = sum(1 for d in diagnoses if d.severity and d.severity.value != "negative")
        elements.append(Paragraph(
            f"Total diagnoses: <b>{total}</b>   |   Positive: <b>{positive}</b>   |   "
            f"Positivity rate: <b>{round(positive / total * 100, 1) if total else 0}%</b>",
            styles["Normal"],
        ))
        elements.append(Spacer(1, 0.5 * cm))

        # Table
        headers = [
            "Date", "Patient Code", "Patient Name", "Facility",
            "Severity", "Parasitaemia %", "Parasites", "Stage (R/T/S/G)",
        ]
        rows = [headers]
        for d in diagnoses:
            r = d.result
            stage_str = ""
            if r:
                stage_str = f"{r.ring_count}/{r.trophozoite_count}/{r.schizont_count}/{r.gametocyte_count}"
            rows.append([
                d.created_at.date().isoformat(),
                d.patient.patient_code if d.patient else "-",
                d.patient.full_name if d.patient else "-",
                d.facility_name or "-",
                d.severity.value if d.severity else "-",
                f"{r.parasitaemia_percent:.2f}" if r else "-",
                str(r.total_parasite_count) if r else "-",
                stage_str,
            ])

        table = Table(rows, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E3F2FD")]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(table)

        doc.build(elements)
        return buffer.getvalue()
