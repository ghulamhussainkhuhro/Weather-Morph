from django.conf import settings
from django.db import models
import uuid
from datetime import timedelta


class Location(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    lat = models.FloatField(null=True, blank=True)
    lon = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name or f"{self.lat}, {self.lon}"


class UserQuery(models.Model):

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="queries", null=True, blank=True
    )
    location = models.ForeignKey(Location, on_delete=models.CASCADE, related_name="queries")
    start_date = models.DateField()                  
    end_date = models.DateField(blank=True, null=True)               
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        # if end_date is not provided, calculate automatically
        if self.start_date and not self.end_date:
            self.end_date = self.start_date + timedelta(days=7)
        super().save(*args, **kwargs)


    def __str__(self):
        return f"Query {self.id} at {self.location}"


class VariableReading(models.Model):
    """
    Holds each variable’s value per day for a given query.
    (This replaces the old static Variables table.)
    """
    VARIABLE_CHOICES = [
        ("T2M", "Temperature at 2m (°C)"),
        ("PRECTOTCORR", "Precipitation Corrected (mm/day)"),
        ("WS10M", "Wind Speed 10m (m/s)"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    query = models.ForeignKey(UserQuery, on_delete=models.CASCADE, related_name="readings")
    variable = models.CharField(max_length=30, choices=VARIABLE_CHOICES)
    date = models.DateField()
    value = models.FloatField()
    unit = models.CharField(max_length=20, blank=True, null=True)

    class Meta:
        unique_together = ("query", "variable", "date")
        ordering = ["date"]

    def __str__(self):
        return f"{self.variable} on {self.date}: {self.value}"


class QueryResult(models.Model):
    """
    Stores raw API response + computed summary.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    query = models.OneToOneField(UserQuery, on_delete=models.CASCADE, related_name="result")
    raw_payload = models.JSONField(help_text="Full raw response from NASA or other source")
    summary = models.JSONField(help_text="Simplified summary, e.g. stats & alerts")
    download_url = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Result for {self.query_id}"

class DailyForecast(models.Model):
    """
    Stores the summarized forecast for each day of a UserQuery.
    e.g., "Day 1 of the query: hot, windy, rainy"
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    query = models.ForeignKey(
        UserQuery,
        on_delete=models.CASCADE,
        related_name="daily_forecasts"
    )

    date = models.DateField()                     # The forecast date (within the query's range)

    title = models.CharField(max_length=100)      # e.g., "Temperature at 2m (°C)"
    value = models.FloatField()                   # e.g., 45.0
    status = models.CharField(max_length=50)      # e.g., "hot", "very windy", "rainy"
    probability = models.FloatField()             # e.g., 75.0  (as percentage)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["date"]
        verbose_name = "Daily Forecast"
        verbose_name_plural = "Daily Forecasts"

    def __str__(self):
        return f"{self.date} - {self.title}: {self.value} ({self.status})"