# Copyright (c) 2025, Renaud Allard <renaud@allard.it>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Tests for the window and sunroof opening binary sensors."""

import pytest
from homeassistant.components.binary_sensor import BinarySensorDeviceClass

from custom_components.cardata.binary_sensor import (
    OPENING_STATUS_DESCRIPTORS,
    OPENING_STATUS_TITLES,
    coerce_binary_value,
    opening_status_to_bool,
)

WINDOW_DESCRIPTOR = "vehicle.cabin.window.row1.driver.status"
DOOR_DESCRIPTOR = "vehicle.cabin.door.row1.driver.isOpen"


class TestOpeningStatusToBool:
    """Tests for the opening enum mapping."""

    def test_closed_is_off(self):
        assert opening_status_to_bool("CLOSED") is False

    @pytest.mark.parametrize("value", ["OPEN", "INTERMEDIATE"])
    def test_open_variants_are_on(self, value):
        assert opening_status_to_bool(value) is True

    def test_value_is_normalized(self):
        assert opening_status_to_bool("  closed  ") is False
        assert opening_status_to_bool("open") is True

    @pytest.mark.parametrize("value", ["UNKNOWN", "NOT_AVAILABLE", "", "INVALID", "TILTED", "PARTIALLY_OPEN"])
    def test_unrecognised_values_return_none(self, value):
        """An unknown enum must not resolve to open, because that raises a false alarm."""
        assert opening_status_to_bool(value) is None

    @pytest.mark.parametrize("value", [None, 1, 0, 12.5, [], {}])
    def test_non_string_returns_none(self, value):
        assert opening_status_to_bool(value) is None

    def test_bool_input_returns_none(self):
        """A boolean is not an opening enum, so it must not be mapped here."""
        assert opening_status_to_bool(True) is None


class TestCoerceBinaryValue:
    """Tests for the shared value coercion used by the binary sensor platform."""

    def test_opening_descriptor_maps_string(self):
        assert coerce_binary_value(WINDOW_DESCRIPTOR, "OPEN") is True
        assert coerce_binary_value(WINDOW_DESCRIPTOR, "CLOSED") is False

    def test_opening_descriptor_rejects_unknown_string(self):
        assert coerce_binary_value(WINDOW_DESCRIPTOR, "SOMETHING_NEW") is None

    def test_boolean_descriptor_passes_through(self):
        assert coerce_binary_value(DOOR_DESCRIPTOR, True) is True
        assert coerce_binary_value(DOOR_DESCRIPTOR, False) is False

    def test_boolean_descriptor_rejects_string(self):
        """Existing behaviour is preserved: non-opening descriptors stay boolean-only."""
        assert coerce_binary_value(DOOR_DESCRIPTOR, "OPEN") is None

    def test_unknown_descriptor_rejects_string(self):
        assert coerce_binary_value("vehicle.something.else", "OPEN") is None


class TestOpeningDescriptorTables:
    """Tests that keep the descriptor tables consistent."""

    def test_every_descriptor_has_a_title(self):
        assert set(OPENING_STATUS_DESCRIPTORS) == set(OPENING_STATUS_TITLES)

    def test_titles_are_distinct(self):
        """Duplicate titles would produce indistinguishable entities."""
        titles = list(OPENING_STATUS_TITLES.values())
        assert len(titles) == len(set(titles))

    def test_every_descriptor_is_reachable_from_a_trigger(self):
        """Home Assistant ships trigger integrations for window and door, but not opening."""
        reachable = {BinarySensorDeviceClass.WINDOW, BinarySensorDeviceClass.DOOR}
        assert set(OPENING_STATUS_DESCRIPTORS.values()) <= reachable
