from rest_framework import serializers
from .models import Location, UserQuery, VariableReading, QueryResult, DailyForecast


class LocationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Location
        fields = ['id', 'lat', 'lon', 'created_at']

class VariableReadingSerializer(serializers.ModelSerializer):
    variable_display = serializers.CharField(source='get_variable_display', read_only=True)

    class Meta:
        model = VariableReading
        fields = ['id', 'variable', 'variable_display', 'date', 'value', 'unit']

class DailyForecastSerializer(serializers.ModelSerializer):
      class Meta:
        model = DailyForecast
        fields = ['date', 'title', 'value', 'status', 'probability']

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

class QueryDetailSerializer(serializers.ModelSerializer):
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
