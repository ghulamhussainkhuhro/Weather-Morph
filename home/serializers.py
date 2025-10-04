# serializers.py
from rest_framework import serializers
from .models import Location, UserQuery, VariableReading, QueryResult, DailyForecast
from datetime import date


# -----------------------------
#  Location
# -----------------------------
class LocationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Location
        fields = ['id', 'lat', 'lon', 'created_at']


# -----------------------------
#  Variable Reading
# -----------------------------
class VariableReadingSerializer(serializers.ModelSerializer):
    variable_display = serializers.CharField(source='get_variable_display', read_only=True)

    class Meta:
        model = VariableReading
        fields = ['id', 'variable', 'variable_display', 'date', 'value', 'unit']


# -----------------------------
#  Daily Forecast
# -----------------------------
class DailyForecastSerializer(serializers.ModelSerializer):
    """
    This serializer will be used to send AI-calculated probability and
    our own computed status for each weather variable per day.
    """

    class Meta:
        model = DailyForecast
        fields = ['date', 'title', 'value', 'status', 'probability']


# -----------------------------
#  User Query (for POST/GET basic info)
# -----------------------------
class UserQuerySerializer(serializers.ModelSerializer):
    location = LocationSerializer()

    class Meta:
        model = UserQuery
        fields = [
            'id',
            'user',
            'location',
            'start_date',
            'end_date',
            'created_at',
            'updated_at'
        ]
        read_only_fields = ['end_date', 'created_at', 'updated_at']


# -----------------------------
#  Query Detail Serializer
# -----------------------------
class QueryDetailSerializer(serializers.ModelSerializer):
    """
    Combines everything for GET endpoint:
      - query info
      - raw variable readings
      - daily forecasts with probability + status
      - summary & download_url
    """

    location = LocationSerializer(read_only=True)
    readings = VariableReadingSerializer(many=True, read_only=True)
    daily_forecasts = DailyForecastSerializer(many=True, read_only=True)
    result = serializers.SerializerMethodField()

    class Meta:
        model = UserQuery
        fields = [
            'id',
            'location',
            'start_date',
            'end_date',
            'readings',
            'daily_forecasts',
            'result'
        ]

    def get_result(self, obj):
        if hasattr(obj, 'result'):
            return {
                'summary': obj.result.summary,
                'download_url': obj.result.download_url
            }
        return None
